import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

from flask import current_app
from groq import Groq
from .util import ApiError

API = "https://generativelanguage.googleapis.com/v1beta/"

DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

RETIRED = re.compile(
    r"^(models/)?gemini-(1\.|2\.0|pro$|pro-vision)",
    re.I
)

_resolved = {}


def _get_key(config_name):
    """Read API key from Flask config or environment."""
    value = (
        current_app.config.get(config_name)
        or os.environ.get(config_name)
        or ""
    )
    return str(value).strip().strip('"').strip("'")


def image_block(data: bytes, media_type: str):
    return {
        "type": "image",
        "source": {
            "media_type": media_type,
            "data": base64.b64encode(data).decode()
        }
    }


def _parts(content):
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]

    parts = []

    for item in content:
        if item.get("type") == "text":
            parts.append({"text": item.get("text", "")})

        elif item.get("type") == "image":
            src = item.get("source", {})
            parts.append({
                "inline_data": {
                    "mime_type": src.get("media_type"),
                    "data": src.get("data")
                }
            })

    return parts


def _http(url, key, payload=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "User-Agent": "FITCONNECT/1.0"
        }
    )

    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def _version(name):
    match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(match.group(1)) if match else 0.0


def discover_model(key):
    data = _http(API + "models?pageSize=200", key)

    names = []

    for model_data in data.get("models", []):
        name = model_data.get("name", "").removeprefix("models/")

        if (
            "generateContent" in
            model_data.get("supportedGenerationMethods", [])
            and re.fullmatch(r"gemini-[\d.]+-flash", name)
        ):
            names.append(name)

    if not names:
        raise ApiError(
            503,
            "ai_model_missing",
            "No usable Gemini model was found."
        )

    return max(names, key=_version)


def _model(key):
    wanted = (
        current_app.config.get("GEMINI_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or DEFAULT_MODEL
    ).strip().removeprefix("models/")

    if RETIRED.match(wanted):
        logging.warning("Gemini model %s may be retired.", wanted)
        wanted = DEFAULT_MODEL

    return _resolved.get(wanted, wanted), wanted


def call_groq(system, messages, max_tokens=900):
    """Backup AI provider using the official Groq Python SDK."""

    key = _get_key("GROQ_API_KEY")

    if not key:
        logging.error("Groq API key is missing.")

        raise ApiError(
            503,
            "groq_not_configured",
            "Groq API key is not configured. Check your .env file."
        )

    model = (
        current_app.config.get("GROQ_MODEL")
        or os.environ.get("GROQ_MODEL")
        or DEFAULT_GROQ_MODEL
    ).strip()

    chat_messages = []

    if system:
        chat_messages.append({
            "role": "system",
            "content": system
        })

    for message in messages:
        content = message.get("content", "")

        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "")
                for item in content
                if item.get("type") == "text"
            )

        chat_messages.append({
            "role": (
                "user" if message.get("role") == "user"
                else "assistant"
            ),
            "content": content
        })

    try:
        client = Groq(api_key=key)

        response = client.chat.completions.create(
            model=model,
            messages=chat_messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="low"
        )

        answer = response.choices[0].message.content

        if not answer:
            raise ValueError("Groq returned an empty response.")

        logging.info("AI response generated using Groq backup.")
        return answer.strip()

    except Exception as error:
        logging.exception("Groq backup request failed.")

        status = getattr(error, "status_code", None)

        if status == 401:
            message = "Groq API key is invalid. Check your GROQ_API_KEY."
        elif status == 404:
            message = "Groq model was not found. Check your GROQ_MODEL."
        elif status == 429:
            message = "Groq rate limit or quota reached. Please try again later."
        else:
            message = "Groq service returned an error. Check the terminal logs."

        raise ApiError(502, "groq_api_error", message)


def call_gemini(system, messages, max_tokens=900):
    """Primary Gemini AI with automatic Groq fallback."""

    key = _get_key("GEMINI_API_KEY")

    if not key:
        logging.warning("Gemini key missing. Switching to Groq.")
        return call_groq(system, messages, max_tokens)

    contents = [
        {
            "role": (
                "user" if message.get("role") == "user"
                else "model"
            ),
            "parts": _parts(message.get("content", ""))
        }
        for message in messages
    ]

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens
        }
    }

    if system:
        payload["systemInstruction"] = {
            "parts": [{"text": system}]
        }

    model, wanted = _model(key)

    for attempt in range(1, 4):
        try:
            data = _http(
                f"{API}models/{model}:generateContent",
                key,
                payload
            )

            answer = "".join(
                part.get("text", "")
                for part in data["candidates"][0]["content"]["parts"]
            ).strip()

            if not answer:
                raise ValueError("Gemini returned an empty response.")

            logging.info("AI response generated using Gemini.")
            return answer

        except urllib.error.HTTPError as error:
            detail = error.read().decode(
                "utf-8", errors="replace"
            )[:1000]

            logging.warning(
                "Gemini API error %s: %s",
                error.code,
                detail
            )

            if error.code in (500, 502, 503, 504):
                if attempt < 3:
                    wait = 2 ** attempt

                    logging.warning(
                        "Gemini unavailable. Retrying in %s seconds.",
                        wait
                    )

                    time.sleep(wait)
                    continue

                logging.warning("Switching from Gemini to Groq.")
                return call_groq(system, messages, max_tokens)

            if error.code == 404 and attempt == 1:
                try:
                    model = _resolved[wanted] = discover_model(key)

                    logging.warning(
                        "Switched Gemini model to %s.",
                        model
                    )
                    continue

                except Exception:
                    logging.exception("Gemini model discovery failed.")
                    return call_groq(system, messages, max_tokens)

            if error.code == 429:
                logging.warning("Gemini quota reached. Switching to Groq.")
                return call_groq(system, messages, max_tokens)

            if error.code in (400, 401, 403):
                logging.warning(
                    "Gemini credentials or request rejected. Trying Groq."
                )
                return call_groq(system, messages, max_tokens)

            raise ApiError(
                502,
                "gemini_api_error",
                "Gemini returned an error. Please try again."
            )

        except ApiError:
            raise

        except Exception:
            logging.exception(
                "Gemini connection failed. Switching to Groq."
            )
            return call_groq(system, messages, max_tokens)

    return call_groq(system, messages, max_tokens)