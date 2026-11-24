import base64
import json
import logging
import re
import urllib.error
import urllib.request
from flask import current_app
from .util import ApiError

API = "https://generativelanguage.googleapis.com/v1beta/"
DEFAULT_MODEL = "gemini-flash-latest"          # Google's alias that always points at the current Flash model
RETIRED = re.compile(r"^(models/)?gemini-(1\.|2\.0|pro$|pro-vision)", re.I)   # shut down by Google
_resolved = {}                                  # configured name -> working name (cached per process)


def image_block(data: bytes, media_type: str):
    return {"type": "image", "source": {"media_type": media_type, "data": base64.b64encode(data).decode()}}


def _parts(content):
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    parts = []
    for item in content:
        if item.get("type") == "text":
            parts.append({"text": item.get("text", "")})
        elif item.get("type") == "image":
            src = item.get("source", {})
            parts.append({"inline_data": {"mime_type": src.get("media_type"), "data": src.get("data")}})
    return parts


def _http(url, key, payload=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"content-type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _version(name):
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def discover_model(key):
    """Asks Google which models this key can use and picks the newest stable Flash model that can generate text."""
    data = _http(API + "models?pageSize=200", key)
    names = []
    for m in data.get("models", []):
        n = m.get("name", "").removeprefix("models/")
        if "generateContent" in m.get("supportedGenerationMethods", []) and re.fullmatch(r"gemini-[\d.]+-flash", n):
            names.append(n)
    if not names:
        raise ApiError(503, "ai_model_missing", "No usable Gemini model was found for this API key. Set GEMINI_MODEL in .env (see https://ai.google.dev/gemini-api/docs/models).")
    return max(names, key=_version)


def _model(key):
    want = (current_app.config.get("GEMINI_MODEL") or DEFAULT_MODEL).strip().removeprefix("models/")
    if RETIRED.match(want):
        logging.warning("GEMINI_MODEL=%s has been shut down by Google; using %s instead. Update .env.", want, DEFAULT_MODEL)
        want = DEFAULT_MODEL
    return _resolved.get(want, want), want


def call_gemini(system, messages, max_tokens=900):
    """Calls the Gemini API from the SERVER (the key never reaches the browser). Picks a working model automatically."""
    key = current_app.config["GEMINI_API_KEY"]
    if not key:
        raise ApiError(503, "ai_unavailable", "AI isn't set up on this server yet. The admin must add a GEMINI_API_KEY.")
    contents = [{"role": "user" if m.get("role") == "user" else "model", "parts": _parts(m.get("content", ""))} for m in messages]
    payload = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}
    model, wanted = _model(key)
    data = None
    for attempt in (1, 2):
        try:
            data = _http(f"{API}models/{model}:generateContent", key, payload)
            break
        except urllib.error.HTTPError as e:
            detail = e.read()[:300]
            if e.code == 404 and attempt == 1:       # model retired or renamed: find a current one and retry once
                try:
                    model = _resolved[wanted] = discover_model(key)
                    logging.warning("Gemini model %s not found; switched to %s. Set GEMINI_MODEL=%s in .env to make this permanent.", wanted, model, model)
                    continue
                except ApiError:
                    raise
                except Exception:
                    logging.exception("Gemini model discovery failed")
            logging.warning("Gemini API error %s: %s", e.code, detail)
            if e.code in (400, 403) and b"API key" in detail:
                raise ApiError(503, "ai_key_invalid", "The Gemini API key was rejected. The admin must check GEMINI_API_KEY in .env.")
            if e.code == 429:
                raise ApiError(429, "ai_busy", "The AI is busy right now (quota reached). Please try again in a minute.")
            raise ApiError(502, "ai_error", "The AI service returned an error. Please try again in a moment.")
        except Exception:
            logging.exception("Gemini request failed")
            raise ApiError(502, "ai_error", "Couldn't reach the AI service. Please try again.")
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError, TypeError):
        raise ApiError(502, "ai_error", "The AI gave an empty answer. Please try again.")