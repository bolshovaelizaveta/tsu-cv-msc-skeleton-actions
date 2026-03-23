from collections import Counter
import numpy as np

FIGHT_CLASSES = {
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "point finger at the other person",
    "falling",
}

HUG_CLASSES = {
    "hugging other person",
    "pat on back of other person",
}

HANDSHAKE_CLASSES = {
    "handshaking",
    "walking towards each other",
}

DANCE_CLASSES = {
    "giving something to other person",
    "touch other person's pocket",
    "walking towards each other",
    "pat on back of other person",
    "hugging other person",
    "cheer up",
}

JUMP_CLASSES = {
    "jump up",
    "hopping (one foot jumping)",
    "cheer up",
}

SMOKING_CANDIDATE_CLASSES = {
    "drink water",
    "brushing teeth",
    "make a phone call/answer phone",
    "wipe face"
}

WALK_CLASSES = {
    "walking apart from each other",
    "staggering",
    "nausea/vomiting condition"
}

SIT_CLASSES = {
    "sitting down",
    "typing on a keyboard"
}

STAND_CLASSES = {
    "standing up",
}


def compute_motion_energy(sequence):
    seq = np.array(sequence)

    if len(seq) < 2:
        return 0.0

    velocity = np.diff(seq[..., :2], axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    return float(speed.mean())


def resolve_target_class(ntu_predictions, last_sequence=None, vlm_action=None):
    counter = Counter(ntu_predictions)

    # --- base grouped scores ---
    punch_cnt = counter.get("punching/slapping other person", 0)
    kick_cnt = counter.get("kicking other person", 0)
    push_cnt = counter.get("pushing other person", 0)
    point_cnt = counter.get("point finger at the other person", 0)
    falling_cnt = counter.get("falling", 0)

    hugging_cnt = counter.get("hugging other person", 0)
    pat_cnt = counter.get("pat on back of other person", 0)
    handshake_cnt = counter.get("handshaking", 0)
    approach_cnt = counter.get("walking towards each other", 0)
    touch_pocket_cnt = counter.get("touch other person's pocket", 0)
    give_cnt = counter.get("giving something to other person", 0)

    cheer_cnt = counter.get("cheer up", 0)
    jump_up_cnt = counter.get("jump up", 0)
    hop_cnt = counter.get("hopping (one foot jumping)", 0)

    aggressive_cnt = punch_cnt + kick_cnt + push_cnt
    real_jump_cnt = jump_up_cnt + hop_cnt

    fight_score = (
        punch_cnt * 3.5 +
        kick_cnt * 3.5 +
        push_cnt * 2.5 +
        point_cnt * 1.0 +
        falling_cnt * 2.0
    )

    hug_score = (
        hugging_cnt * 1.5 +
        pat_cnt * 1.2
    )

    handshake_score = (
        handshake_cnt * 3.0 +
        approach_cnt * 1.2
    )

    # усиливаем dance относительно исходной версии
    dance_score = (
        give_cnt * 2.0 +
        touch_pocket_cnt * 2.2 +
        approach_cnt * 1.5 +
        pat_cnt * 1.2 +
        hugging_cnt * 0.8 +
        cheer_cnt * 2.5
    )

    # jump делаем строже: cheer up не должен доминировать
    jump_score = (
        jump_up_cnt * 10.0 +
        hop_cnt * 10.0 +
        cheer_cnt * 1.5
    )

    # walking towards each other больше не кормит walk
    walk_score = (
        counter.get("walking apart from each other", 0) * 4.0 +
        counter.get("staggering", 0) * 3.0 +
        counter.get("nausea/vomiting condition", 0) * 2.0
    )

    sit_score = (
        counter.get("sitting down", 0) * 5.0 +
        counter.get("typing on a keyboard", 0) * 1.0 +
        counter.get("playing with phone/tablet", 0) * 1.0 +
        counter.get("reading", 0) * 1.0 +
        counter.get("writing", 0) * 1.0
    )

    stand_score = sum(counter[c] for c in STAND_CLASSES if c in counter)

    smoking_candidate_score = (
        counter.get("drink water", 0) * 4.0 +
        counter.get("make a phone call/answer phone", 0) * 4.0 +
        counter.get("brushing teeth", 0) * 3.0 +
        counter.get("wipe face", 0) * 3.0 +
        counter.get("touch head (headache)", 0) * 2.0
    )

    scores = {
        "fight": fight_score,
        "hug": hug_score,
        "handshake": handshake_score,
        "dance": dance_score,
        "jump": jump_score,
        "walk": walk_score,
        "sit": sit_score,
        "stand": stand_score,
        "smoking_candidate": smoking_candidate_score * 1.2
    }

    # --- motion-based correction ---
    if last_sequence is not None:
        motion = compute_motion_energy(last_sequence)

        if motion > 0.08:
            scores["fight"] *= 1.15
            scores["jump"] *= 1.15
            scores["dance"] *= 1.15
            scores["walk"] *= 1.05
            scores["smoking_candidate"] *= 0.5
        else:
            scores["dance"] *= 1.20
            scores["hug"] *= 1.05
            scores["sit"] *= 1.3
            scores["smoking_candidate"] *= 1.2

        # --- handshake rescue ---
    social_contact_cnt = handshake_cnt + approach_cnt + pat_cnt + give_cnt + hugging_cnt
    aggressive_cnt = punch_cnt + kick_cnt + push_cnt
    aggressive_ratio = aggressive_cnt / max(social_contact_cnt, 1)

    # handshake rescue
    if approach_cnt >= 20 and social_contact_cnt >= 35 and real_jump_cnt <= 3:
        scores["handshake"] += 180.0

    if approach_cnt >= 40 and approach_cnt > aggressive_cnt:
        scores["handshake"] += 120.0
        scores["fight"] *= 0.55
        scores["dance"] *= 0.85

    if social_contact_cnt >= 30 and aggressive_ratio < 0.45:
        scores["fight"] *= 0.60

    if approach_cnt >= 25 and social_contact_cnt >= 35 and aggressive_cnt < approach_cnt:
        scores["dance"] *= 0.80

    if handshake_cnt + approach_cnt >= 50 and real_jump_cnt <= 2:
        scores["handshake"] += 100.0
        scores["jump"] *= 0.5

    total_frames = len(ntu_predictions)
    if total_frames > 0:
        # только реальные jump-классы, без cheer up
        real_jump_percent = (jump_up_cnt + hop_cnt) / total_frames
        cheer_percent = cheer_cnt / total_frames

        if real_jump_percent > 0.15:
            scores["jump"] += 250.0
            scores["dance"] *= 0.75

        # cheer up при отсутствии реальных прыжков скорее поддерживает dance
        if cheer_percent > 0.15 and real_jump_cnt == 0:
            scores["dance"] += 80.0
            scores["jump"] *= 0.6

    # friendly interaction boost for dance
    friendly_total = (
        give_cnt +
        touch_pocket_cnt +
        approach_cnt +
        pat_cnt +
        hugging_cnt +
        cheer_cnt
    )

    if friendly_total > aggressive_cnt * 1.5:
        scores["dance"] += 60.0

    # optional VLM override
    if vlm_action:
        v_act = str(vlm_action).lower()

        if any(s in v_act for s in ["smoke", "smoking", "cigarette", "vape"]):
            scores["smoking_candidate"] += 5000
        elif any(s in v_act for s in ["tug", "war", "rope", "pulling"]):
            scores["tug_of_war"] = 5000
        elif any(s in v_act for s in ["rally", "meeting", "protest", "crowd", "gathering"]):
            scores["meeting"] = 5000
        elif any(s in v_act for s in ["circle", "triangle", "formation", "dance", "round", "horovod"]):
            scores["circle_triangle"] = 5000

    final_class = max(scores, key=scores.get)
    return final_class, scores


def map_ntu_to_target(ntu_class):
    if ntu_class in JUMP_CLASSES:
        return "jump"
    if ntu_class in FIGHT_CLASSES:
        return "fight"
    if ntu_class in HUG_CLASSES:
        return "hug"
    if ntu_class in HANDSHAKE_CLASSES:
        return "handshake"
    if ntu_class in DANCE_CLASSES:
        return "dance"
    if ntu_class in WALK_CLASSES:
        return "walk"
    if ntu_class in SIT_CLASSES:
        return "sit"
    if ntu_class in STAND_CLASSES:
        return "stand"
    if ntu_class in SMOKING_CANDIDATE_CLASSES:
        return "smoking_candidate"

    return None