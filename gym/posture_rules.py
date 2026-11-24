
"""FITCONNECT exercise-specific posture rules.
Rule-based landmark feedback. Single-image estimates only.
"""

import math

IDX = {
    "L": {"shoulder": 11, "elbow": 13, "wrist": 15,
          "hip": 23, "knee": 25, "ankle": 27},
    "R": {"shoulder": 12, "elbow": 14, "wrist": 16,
          "hip": 24, "knee": 26, "ankle": 28}
}

VIS = 0.60
CONF = 0.70

DISCLAIMER = (
    "AI posture estimate from landmark detection. A single image cannot"
    "evaluate complete movement or diagnose injuries. Stop if anything hurts"
    "and consult a qualified coach."
)

# Exercise catalogue: Squat intentionally excluded.
BODY_PART_EXERCISES = {
    "Chest": [
        "Bench press", "Incline dumbbell press", "Chest fly",
        "Push-up", "Cable crossover", "Pec deck"
    ],
    "Back": [
        "Lat pulldown", "Seated cable row", "Bent-over row",
        "Single-arm dumbbell row", "Deadlift", "Pull-up"
    ],
    "Shoulders": [
        "Shoulder press", "Lateral raise", "Front raise",
        "Reverse fly", "Arnold press", "Face pull"
    ],
    "Biceps": [
        "Biceps curl", "Hammer curl", "Preacher curl",
        "Concentration curl", "Cable curl"
    ],
    "Triceps": [
        "Triceps pushdown", "Overhead triceps extension",
        "Skull crusher", "Bench dip", "Close-grip push-up"
    ],
    "Legs": [
        "Lunge", "Leg press", "Leg extension", "Leg curl",
        "Romanian deadlift", "Calf raise"
    ],
    "Abs": [
        "Plank", "Crunch", "Bicycle crunch", "Leg raise",
        "Russian twist", "Mountain climber"
    ]
}

ALIASES = {
    "chestfly": "chest_fly",
    "bench_fly": "chest_fly",
    "push_up": "pushup",
    "push_ups": "pushup",
    "bicep_curl": "biceps_curl",
    "tricep_pushdown": "triceps_pushdown",
    "lat_pulldown": "lat_pulldown",
    "single_arm_dumbbell_row": "single_arm_dumbbell_row"
}


def normalise(exercise):
    ex = str(exercise).strip().lower()
    ex = ex.replace("-", "_").replace(" ", "_")
    return ALIASES.get(ex, ex)


# Required visible body landmarks for each exercise.
NEED = {}

for exercises in BODY_PART_EXERCISES.values():
    for exercise in exercises:
        key = normalise(exercise)

        if key in (
            "lunge", "leg_press", "leg_extension",
            "leg_curl", "romanian_deadlift", "calf_raise"
        ):
            NEED[key] = ["shoulder", "hip", "knee", "ankle"]

        elif key in (
            "plank", "mountain_climber", "pushup",
            "bench_dip", "close_grip_push_up"
        ):
            NEED[key] = ["shoulder", "elbow", "wrist", "hip", "ankle"]

        elif key in (
            "crunch", "bicycle_crunch", "leg_raise",
            "russian_twist"
        ):
            NEED[key] = ["shoulder", "hip", "knee"]

        elif key in (
            "lateral_raise", "front_raise", "reverse_fly",
            "shoulder_press", "arnold_press", "face_pull"
        ):
            NEED[key] = ["shoulder", "elbow", "wrist", "hip"]

        else:
            NEED[key] = ["shoulder", "elbow", "wrist", "hip"]


# Kept for compatibility with posture.py imports.
_EXTRA_NEED = dict(NEED)

TIPS = {
    ex: (
        "Keep the full required body parts visible, use good lighting "
        "and position the camera at your side."
    )
    for ex in NEED
}


def _ang(a, b, c):
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*u) * math.hypot(*v)

    if not n:
        return None

    value = (u[0] * v[0] + u[1] * v[1]) / n
    return math.degrees(math.acos(max(-1, min(1, value))))


def _fail(ex, status, conf, missing, message):
    return {
        "exercise": ex,
        "status": status,
        "confidence": round(conf, 2),
        "missing_joints": missing,
        "message": message,
        "issues": [],
        "corrections": [],
        "observations": [],
        "metrics": {},
        "overlay": {"points": [], "angles": []},
        "single_image": True,
        "disclaimer": DISCLAIMER
    }


def analyze(ex, lms, w, h):
    ex = normalise(ex)

    if ex not in NEED:
        raise ValueError(
            "This exercise is not available in the posture exercise list."
        )

    if not lms or len(lms) < 29:
        return _fail(
            ex, "insufficient_visibility", 0, ["body"],
            "No complete pose was detected. " + TIPS[ex]
        )

    need = NEED[ex]
    best = None

    for side in ("L", "R"):
        visibility = {}

        for joint in need:
            index = IDX[side][joint]
            visibility[joint] = (
                float(lms[index][2])
                if index < len(lms) else 0
            )

        score = sum(visibility.values()) / len(visibility)

        if best is None or score > best[0]:
            best = (score, side, visibility)

    conf, side, visibility = best

    missing = [
        joint for joint in need
        if visibility[joint] < VIS
    ]

    if missing:
        return _fail(
            ex, "insufficient_visibility", conf, missing,
            "These body parts are not clearly visible: "
            + ", ".join(missing) + ". " + TIPS[ex]
        )

    if conf < CONF:
        return _fail(
            ex, "low_confidence", conf, [],
            "Detection confidence is low. Improve lighting and camera "
            "position. " + TIPS[ex]
        )

    P = {}
    N = {}

    for joint in need:
        index = IDX[side][joint]
        P[joint] = (
            float(lms[index][0]) * w,
            float(lms[index][1]) * h
        )
        N[joint] = (
            float(lms[index][0]),
            float(lms[index][1])
        )

    issues = []
    corrections = []
    observations = []
    metrics = {}
    angles = []

    def add(ok, title, detail, correction=None):
        issues.append({
            "severity": "ok" if ok else "warn",
            "title": title,
            "detail": detail
        })

        if not ok and correction:
            corrections.append(correction)

    def measure(a, b, c, label):
        value = _ang(P[a], P[b], P[c])

        if value is not None:
            metrics[label] = round(value)
            angles.append({
                "label": f"{round(value)}° {label}",
                "x": N[b][0],
                "y": N[b][1]
            })
            observations.append(
                f"{label.replace('_', ' ').title()} angle: "
                f"{round(value)}° (estimate)"
            )

        return value

    # --------------------------------------------------
    # LEGS: Lunge and leg exercises
    # --------------------------------------------------
    if ex in (
        "lunge", "leg_press", "leg_extension",
        "leg_curl", "romanian_deadlift", "calf_raise"
    ):
        knee = measure("hip", "knee", "ankle", "knee")

        if knee is not None:
            if knee < 65:
                add(
                    False, "Deep knee bend",
                    "The knee is strongly bent in this image.",
                    "Use a comfortable range of motion and avoid forcing the position."
                )
            elif knee > 165:
                add(
                    True, "Knee nearly extended",
                    "The knee appears relatively straight in this image."
                )
            else:
                add(
                    True, "Knee position detected",
                    "Knee alignment is visible. Full movement cannot be judged from one image."
                )

        torso = measure("shoulder", "hip", "knee", "torso")

        observations.append(
            "Keep your movement controlled and avoid forcing your joints."
        )

        if torso is not None and torso < 90:
            add(
                False, "Torso position",
                "Your upper body appears strongly folded.",
                "Keep your back controlled and use a comfortable range of motion."
            )

    # --------------------------------------------------
    # UPPER BODY: Chest, back, shoulders, arms
    # --------------------------------------------------
    elif "shoulder" in need and "elbow" in need and "wrist" in need:
        elbow = measure("shoulder", "elbow", "wrist", "elbow")

        if elbow is not None:
            if ex in ("biceps_curl", "hammer_curl",
                      "preacher_curl", "concentration_curl",
                      "cable_curl"):
                observations.append(
                    "For curls, keep your upper arm controlled while bending the elbow."
                )
                if elbow < 25:
                    add(
                        False, "Very closed elbow",
                        "Your elbow is strongly bent in this image.",
                        "Use a controlled range and avoid forcing the joint."
                    )
                else:
                    add(
                        True, "Elbow detected",
                        "Your elbow position is visible. Movement control needs video analysis."
                    )

            elif ex in ("lateral_raise", "front_raise", "reverse_fly"):
                observations.append(
                    "For raises, avoid swinging your body or shrugging your shoulders."
                )
                add(
                    True, "Arm position detected",
                    "Shoulder and elbow landmarks are visible."
                )

            elif ex in ("triceps_pushdown", "overhead_triceps_extension",
                        "skull_crusher"):
                observations.append(
                    "Keep your upper arms controlled and move mainly through the elbows."
                )
                add(
                    True, "Elbow position detected",
                    "Check that your elbows remain controlled during the movement."
                )

            else:
                observations.append(
                    "Keep wrists controlled and avoid forcing your shoulders or elbows."
                )
                add(
                    True, "Upper-body landmarks detected",
                    "Arm position is visible. Full exercise technique needs movement analysis."
                )

        if ex in ("chest_fly", "pec_deck", "cable_crossover"):
            observations.append(
                "Keep a slight elbow bend and avoid forcing your arms too far behind your body."
            )

        if ex in ("bent_over_row", "single_arm_dumbbell_row",
                  "seated_cable_row", "deadlift",
                  "romanian_deadlift", "lat_pulldown",
                  "pull_up", "face_pull"):
            observations.append(
                "Keep your back controlled and avoid jerking the weight."
            )

    # --------------------------------------------------
    # CORE: Abs and plank exercises
    # --------------------------------------------------
    elif ex in (
        "plank", "crunch", "bicycle_crunch",
        "leg_raise", "russian_twist", "mountain_climber"
    ):
        if ex == "plank":
            body = measure("shoulder", "hip", "ankle", "body_line")

            if body is not None:
                if body >= 165:
                    add(
                        True, "Body alignment",
                        "Your shoulder, hip and ankle appear roughly aligned."
                    )
                else:
                    add(
                        False, "Body alignment needs attention",
                        "Your body line appears bent in this image.",
                        "Brace your core and keep your shoulders, hips and ankles aligned."
                    )

            observations.append(
                "Keep your elbows under your shoulders when performing an elbow plank."
            )

        else:
            measure("shoulder", "hip", "knee", "torso")

            observations.append(
                "Keep your core engaged and avoid pulling your neck during abdominal exercises."
            )

            add(
                True, "Core landmarks detected",
                "Your body position is visible. Complete movement quality needs video analysis."
            )

    # Fallback for any registered exercise.
    if not issues:
        add(
            True, "Posture landmarks detected",
            "Required body landmarks are visible. Follow controlled movement technique."
        )

    points = [
        {
            "id": i,
            "x": float(lms[i][0]),
            "y": float(lms[i][1])
        }
        for i in range(11, 29)
        if i < len(lms) and float(lms[i][2]) >= VIS
    ]

    return {
        "exercise": ex,
        "status": "ok",
        "confidence": round(conf, 2),
        "missing_joints": [],
        "message": None,
        "observations": observations,
        "issues": issues,
        "corrections": corrections,
        "metrics": metrics,
        "overlay": {
            "points": points,
            "angles": angles
        },
        "single_image": True,
        "disclaimer": DISCLAIMER
    }
