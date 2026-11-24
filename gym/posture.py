


import json
import re
from flask import Blueprint, jsonify, request, current_app
from . import pose
from .db import fetch, fetch1, run
from .posture_rules import analyze, NEED, _EXTRA_NEED
from .security import subscription_required, current_user
from .util import ApiError, RL

bp = Blueprint("posture", __name__, url_prefix="/api/posture")
MAX = 8 * 1024 * 1024


@bp.post("/analyze")
@subscription_required
def analyze_photo():
    u = current_user()
    RL.check(f"posture:{u['id']}", 30, 3600)

    # The UI sends the exact exercise name from Workout Logbook.
    raw_ex = (request.form.get("exercise") or "").strip()
    body_part = (request.form.get("body_part") or "").strip().lower()
    if not raw_ex:
        raise ApiError(422, "invalid", "Choose an exercise from the selected body-part list.", "exercise")

    import re
    key = re.sub(r"[^a-z0-9]+", " ", raw_ex.lower()).strip()
    aliases = {
        "push up": "pushup", "close grip push up": "close-grip push-up",
        "pullup": "pull-up", "bent over row": "bent-over row",
        "single arm dumbbell row": "single-arm dumbbell row",
        "bicep curl": "biceps curl", "tricep pushdown": "triceps pushdown",
    }
    key = aliases.get(key, key)
    canonical = {re.sub(r"[^a-z0-9]+", " ", name.lower()).strip(): name for name in NEED}
    ex = canonical.get(key, key)

    # If the Logbook has an additional exercise name, use the selected body-part
    # family to choose relevant joints and run conservative landmark checks.
    if ex not in NEED:
        family_joints = {
            "chest": ["shoulder", "elbow", "wrist", "hip"],
            "back": ["shoulder", "elbow", "wrist", "hip"],
            "shoulders": ["shoulder", "elbow", "wrist", "hip"],
            "biceps": ["shoulder", "elbow", "wrist"],
            "triceps": ["shoulder", "elbow", "wrist"],
            "legs": ["shoulder", "hip", "knee", "ankle"],
            "abs": ["shoulder", "hip", "knee", "ankle"],
        }
        if body_part not in family_joints:
            raise ApiError(422, "invalid", "Choose a body part and exercise from Workout Logbook.", "exercise")
        NEED[ex] = family_joints[body_part]
        _EXTRA_NEED[ex] = family_joints[body_part]

    f = request.files.get("image")
    if not f:
        raise ApiError(422, "invalid", "Upload a JPG or PNG photo.", "image")

    data = f.read(MAX + 1)
    if len(data) > MAX:
        raise ApiError(413, "too_large", "Photo must be under 8 MB.", "image")

    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        raise ApiError(422, "invalid", "Please upload a JPG or PNG photo (PDFs aren't supported - save the photo as JPG).", "image")

    # The photo itself is never stored. Known exercises use their exercise rules;
    # other Logbook entries use conservative checks for their selected body-part family.
    lms, w, h = pose.detect(data, current_app.config["POSE_MODEL_PATH"])
    res = analyze(ex, lms, w, h)

    # Recommend only videos that match the entered exercise name.
    has_problem = (
        res.get("status") != "ok"
        or any(i.get("severity") == "warn" for i in res.get("issues", []))
    )
    if has_problem:
        def words(value):
            return re.findall(r"[a-z0-9]+", (value or "").lower())

        exercise_words = words(ex)
        matches = []
        if exercise_words:
            for video in fetch("SELECT id,title,category,trainer_name,instructions,thumb FROM videos ORDER BY id DESC"):
                searchable = set(words(video.get("title", "") + " " + video.get("instructions", "")))
                if any(word in searchable for word in exercise_words):
                    matches.append({
                        "id": video["id"],
                        "title": video["title"],
                        "category": video["category"],
                        "trainer_name": video["trainer_name"],
                        "has_thumb": bool(video.get("thumb"))
                    })
                if len(matches) >= 3:
                    break
        res["recommended_videos"] = matches
