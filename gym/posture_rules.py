"""FITCONNECT exercise-specific posture rules.
Rule-based landmark feedback. Single-image estimates only.
"""

import math

IDX = {
    "L": {
        "shoulder": 11, "elbow": 13, "wrist": 15,
        "hip": 23, "knee": 25, "ankle": 27,
    },
    "R": {
        "shoulder": 12, "elbow": 14, "wrist": 16,
        "hip": 24, "knee": 26, "ankle": 28,
    },
}

VIS = 0.60
CONF = 0.70

DISCLAIMER = (
    "AI posture estimate from landmark detection. A single image cannot "
    "evaluate complete movement or diagnose injuries. Stop if anything hurts "
    "and consult a qualified coach."
)

BODY_PART_EXERCISES = {
    "Chest": [
        "Bench press", "Incline dumbbell press", "Chest fly",
        "Push-up", "Cable crossover", "Pec deck",
    ],
    "Back": [
        "Lat pulldown", "Seated cable row", "Bent-over row",
        "Single-arm dumbbell row", "Deadlift", "Pull-up",
    ],
    "Shoulders": [
        "Shoulder press", "Lateral raise", "Front raise",
        "Reverse fly", "Arnold press", "Face pull",
    ],
    "Biceps": [
        "Biceps curl", "Hammer curl", "Preacher curl",
        "Concentration curl", "Cable curl",
    ],
    "Triceps": [
        "Triceps pushdown", "Overhead triceps extension",
        "Skull crusher", "Bench dip", "Close-grip push-up",
    ],
    "Legs": [
        "Squat", "Lunge", "Leg press", "Leg extension",
        "Leg curl", "Romanian deadlift", "Calf raise",
    ],
    "Abs": [
        "Plank", "Crunch", "Bicycle crunch", "Leg raise",
        "Russian twist", "Mountain climber",
    ],
}

ALIASES = {
    "chestfly": "chest_fly",
    "bench_fly": "chest_fly",
    "push_up": "pushup",
    "push_ups": "pushup",
    "close_grip_pushup": "close_grip_push_up",
    "pullup": "pull_up",
    "bent_over_row": "bent_over_row",
    "single_arm_dumbbell_row": "single_arm_dumbbell_row",
    "bicep_curl": "biceps_curl",
    "tricep_pushdown": "triceps_pushdown",
}


def normalise(exercise):
    ex = str(exercise).strip().lower()
    ex = ex.replace("-", "_").replace(" ", "_")
    return ALIASES.get(ex, ex)


# Required visible landmarks for each registered exercise.
NEED = {
    "bench_press": ["shoulder", "elbow", "wrist", "hip"],
    "incline_dumbbell_press": ["shoulder", "elbow", "wrist", "hip"],
    "chest_fly": ["shoulder", "elbow", "wrist", "hip"],
    "pushup": ["shoulder", "elbow", "wrist", "hip", "ankle"],
    "cable_crossover": ["shoulder", "elbow", "wrist", "hip"],
    "pec_deck": ["shoulder", "elbow", "wrist", "hip"],

    "lat_pulldown": ["shoulder", "elbow", "wrist", "hip"],
    "seated_cable_row": ["shoulder", "elbow", "wrist", "hip"],
    "bent_over_row": ["shoulder", "elbow", "wrist", "hip", "knee"],
    "single_arm_dumbbell_row": ["shoulder", "elbow", "wrist", "hip", "knee"],
    "deadlift": ["shoulder", "hip", "knee", "ankle"],
    "pull_up": ["shoulder", "elbow", "wrist", "hip"],

    "shoulder_press": ["shoulder", "elbow", "wrist", "hip"],
    "lateral_raise": ["shoulder", "elbow", "wrist", "hip"],
    "front_raise": ["shoulder", "elbow", "wrist", "hip"],
    "reverse_fly": ["shoulder", "elbow", "wrist", "hip"],
    "arnold_press": ["shoulder", "elbow", "wrist", "hip"],
    "face_pull": ["shoulder", "elbow", "wrist", "hip"],

    "biceps_curl": ["shoulder", "elbow", "wrist"],
    "hammer_curl": ["shoulder", "elbow", "wrist"],
    "preacher_curl": ["shoulder", "elbow", "wrist"],
    "concentration_curl": ["shoulder", "elbow", "wrist"],
    "cable_curl": ["shoulder", "elbow", "wrist"],

    "triceps_pushdown": ["shoulder", "elbow", "wrist"],
    "overhead_triceps_extension": ["shoulder", "elbow", "wrist"],
    "skull_crusher": ["shoulder", "elbow", "wrist"],
    "bench_dip": ["shoulder", "elbow", "wrist", "hip"],
    "close_grip_push_up": ["shoulder", "elbow", "wrist", "hip", "ankle"],

    "squat": ["shoulder", "hip", "knee", "ankle"],
    "lunge": ["shoulder", "hip", "knee", "ankle"],
    "leg_press": ["shoulder", "hip", "knee", "ankle"],
    "leg_extension": ["hip", "knee", "ankle"],
    "leg_curl": ["hip", "knee", "ankle"],
    "romanian_deadlift": ["shoulder", "hip", "knee", "ankle"],
    "calf_raise": ["hip", "knee", "ankle"],

    "plank": ["shoulder", "elbow", "hip", "ankle"],
    "crunch": ["shoulder", "hip", "knee"],
    "bicycle_crunch": ["shoulder", "hip", "knee"],
    "leg_raise": ["shoulder", "hip", "knee", "ankle"],
    "russian_twist": ["shoulder", "hip", "knee"],
    "mountain_climber": ["shoulder", "elbow", "wrist", "hip", "ankle"],
}

# Kept for compatibility with posture.py imports.
_EXTRA_NEED = dict(NEED)

TIPS = {
    ex: (
        "Keep all required joints visible in the frame, use good lighting, "
        "and position the camera so your side or working angle is clear."
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
    return math.degrees(math.acos(max(-1.0, min(1.0, value))))


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
        "disclaimer": DISCLAIMER,
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
            "No complete pose was detected. " + TIPS.get(ex, "")
        )

    need = NEED[ex]
    best = None

    for side in ("L", "R"):
        visibility = {}
        for joint in need:
            index = IDX[side][joint]
            visibility[joint] = float(lms[index][2]) if index < len(lms) else 0.0

        score = sum(visibility.values()) / len(visibility)
        if best is None or score > best[0]:
            best = (score, side, visibility)

    conf, side, visibility = best
    missing = [joint for joint in need if visibility[joint] < VIS]

    if missing:
        return _fail(
            ex, "insufficient_visibility", conf, missing,
            "These body parts are not clearly visible: "
            + ", ".join(missing) + ". " + TIPS.get(ex, "")
        )

    if conf < CONF:
        return _fail(
            ex, "low_confidence", conf, [],
            "Detection confidence is low. Improve lighting and camera "
            "position. " + TIPS.get(ex, "")
        )

    P = {}
    N = {}
    for joint in need:
        index = IDX[side][joint]
        P[joint] = (float(lms[index][0]) * w, float(lms[index][1]) * h)
        N[joint] = (float(lms[index][0]), float(lms[index][1]))

    issues = []
    corrections = []
    observations = []
    metrics = {}
    angles = []

    def add(ok, title, detail, correction=None):
        issues.append({
            "severity": "ok" if ok else "warn",
            "title": title,
            "detail": detail,
        })
        if not ok and correction:
            corrections.append(correction)

    def measure(a, b, c, label):
        if a not in P or b not in P or c not in P:
            return None
        value = _ang(P[a], P[b], P[c])
        if value is not None:
            metrics[label] = round(value)
            angles.append({
                "label": f"{round(value)}° {label}",
                "x": N[b][0],
                "y": N[b][1],
            })
            observations.append(
                f"{label.replace('_', ' ').title()} angle: {round(value)}° (estimate)"
            )
        return value

    # --------------------------------------------------
    # LEGS
    # --------------------------------------------------
    if ex in (
        "squat", "lunge", "leg_press", "leg_extension",
        "leg_curl", "romanian_deadlift", "calf_raise"
    ):
        knee = measure("hip", "knee", "ankle", "knee")
        if knee is not None:
            if knee < 60:
                add(
                    False, "Deep knee bend",
                    "Knee flexion appears deep in this single-frame estimate.",
                    "Keep movement controlled and avoid forcing deep joint flexion."
                )
            elif knee > 165:
                add(
                    True, "Knee nearly extended",
                    "Knee joint appears nearly straight in this frame."
                )
            else:
                add(
                    True, "Knee angle detected",
                    "Knee alignment is visible in this frame."
                )

        if "shoulder" in P and "hip" in P and "knee" in P:
            torso = measure("shoulder", "hip", "knee", "torso")
            if torso is not None and torso < 85:
                add(
                    False, "Forward torso lean",
                    "Upper body shows a strong forward angle in this frame.",
                    "Brace your core and avoid excessive forward trunk lean."
                )

        observations.append(
            "Maintain steady control and balanced load distribution."
        )

    # --------------------------------------------------
    # CORE / ABS
    # --------------------------------------------------
    elif ex in (
        "plank", "crunch", "bicycle_crunch",
        "leg_raise", "russian_twist", "mountain_climber"
    ):
        if ex == "plank":
            ref_joint = "elbow" if "elbow" in P else "wrist"
            body = measure("shoulder", "hip", "ankle", "body_line")
            if body is not None:
                if body >= 165:
                    add(
                        True, "Body line aligned",
                        "Shoulder, hip, and ankle alignment looks steady."
                    )
                else:
                    add(
                        False, "Body alignment drift",
                        "Shoulder, hip, and ankle line appears bent in this frame.",
                        "Engage your core and avoid letting your hips sag or pike."
                    )
            if ref_joint in P:
                observations.append(
                    f"Keep your {ref_joint} stacked under your shoulder for support."
                )
        else:
            torso = measure("shoulder", "hip", "knee", "torso")
            if torso is not None:
                add(
                    True, "Core position detected",
                    "Trunk and hip landmarks are visible in this frame."
                )
            observations.append(
                "Control the movement with your abs rather than pulling with your neck."
            )

    # --------------------------------------------------
    # UPPER BODY: Chest, Back, Shoulders, Biceps, Triceps
    # --------------------------------------------------
    else:
        if "shoulder" in P and "elbow" in P and "wrist" in P:
            elbow = measure("shoulder", "elbow", "wrist", "elbow")
            if elbow is not None:
                if ex in (
                    "biceps_curl", "hammer_curl", "preacher_curl",
                    "concentration_curl", "cable_curl"
                ):
                    if elbow < 25:
                        add(
                            False, "Extreme elbow flexion",
                            "Elbow angle is tightly closed in this frame.",
                            "Use a controlled range of motion without forcing the joint."
                        )
                    else:
                        add(
                            True, "Elbow flexion detected",
                            "Elbow angle is visible in this frame."
                        )
                    observations.append(
                        "Keep your upper arm steady and avoid swinging the torso."
                    )

                elif ex in (
                    "triceps_pushdown", "overhead_triceps_extension",
                    "skull_crusher", "bench_dip", "close_grip_push_up"
                ):
                    add(
                        True, "Elbow extension angle detected",
                        "Elbow position is visible in this frame."
                    )
                    observations.append(
                        "Keep elbows controlled and avoid flaring or drifting outward."
                    )

                elif ex in ("lateral_raise", "front_raise", "reverse_fly"):
                    add(
                        True, "Arm position detected",
                        "Shoulder and elbow landmarks are visible in this frame."
                    )
                    observations.append(
                        "Raise smoothly without shrugging your shoulders or swinging."
                    )

                else:
                    add(
                        True, "Upper-body alignment detected",
                        "Key upper-body joints are visible in this frame."
                    )
                    observations.append(
                        "Keep wrists neutral and control both concentric and eccentric phases."
                    )

        if "shoulder" in P and "hip" in P and "knee" in P:
            measure("shoulder", "hip", "knee", "torso")

        if ex in (
            "bent_over_row", "single_arm_dumbbell_row", "seated_cable_row",
            "deadlift", "lat_pulldown", "pull_up", "face_pull"
        ):
            observations.append(
                "Keep your spine neutral and avoid jerking or rounding your back."
            )

    if not issues:
        add(
            True, "Landmarks detected",
            "Required joints are visible for this exercise estimate."
        )

    points = [
        {
            "id": i,
            "x": float(lms[i][0]),
            "y": float(lms[i][1]),
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
            "angles": angles,
        },
        "single_image": True,
        "disclaimer": DISCLAIMER,
    }