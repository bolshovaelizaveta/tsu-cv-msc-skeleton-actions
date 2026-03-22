from collections import Counter
import numpy as np

FIGHT_CLASSES = {
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "hugging other person",
    "pat on back of other person",
    "touch other person's pocket",
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
    "wipe face",
    "touch head (headache)",    
    "touch neck (neckache)",     
    "take off glasses",          
    "wear on glasses"    
}

WALK_CLASSES = {
    "walking apart from each other",
    "staggering"
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

    # --- базовые суммы по группам ---
    fight_score = (
        counter.get("punching/slapping other person", 0) * 3.0 +
        counter.get("kicking other person", 0) * 3.0 +
        counter.get("pushing other person", 0) * 2.0 +
        counter.get("hugging other person", 0) * 1.0 +
        counter.get("pat on back of other person", 0) * 1.0 +
        counter.get("touch other person's pocket", 0) * 1.0 +
        counter.get("point finger at the other person", 0) * 1.0 +
        counter.get("falling", 0) * 2.0
    )

    hug_score = sum(counter[c] for c in HUG_CLASSES if c in counter)
    handshake_score = counter.get("handshaking", 0) * 3.0 + counter.get("walking towards each other", 0) * 1.0

    dance_score = (
        counter.get("giving something to other person", 0) * 1.5 +
        counter.get("touch other person's pocket", 0) * 1.5 +
        counter.get("walking towards each other", 0) * 1.0 +
        counter.get("pat on back of other person", 0) * 1.0 +
        counter.get("hugging other person", 0) * 1.0 +
        counter.get("cheer up", 0) * 1.5
    )

    jump_score = sum(counter[c] for c in JUMP_CLASSES if c in counter)
    walk_score = sum(counter[c] for c in WALK_CLASSES if c in counter)
    sit_score = sum(counter[c] for c in SIT_CLASSES if c in counter)
    stand_score = sum(counter[c] for c in STAND_CLASSES if c in counter)
    smoking_candidate_score = sum(counter[c] for c in SMOKING_CANDIDATE_CLASSES if c in counter)

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
            scores["fight"] *= 1.35
            scores["jump"] *= 1.15
            scores["dance"] *= 0.90
            scores["smoking_candidate"] *= 0.5
        else:
            scores["dance"] *= 1.15
            scores["hug"] *= 1.05
            scores["smoking_candidate"] *= 1.2

    # --- rule-based overrides ---
    punch_cnt = counter.get("punching/slapping other person", 0)
    kick_cnt = counter.get("kicking other person", 0)
    push_cnt = counter.get("pushing other person", 0)
    aggressive_cnt = punch_cnt + kick_cnt + push_cnt

    if aggressive_cnt > 20:
        scores["fight"] += aggressive_cnt * 2.0
    if aggressive_cnt > 10 and scores["fight"] > scores["hug"] * 0.7:
        scores["fight"] += 50
    if aggressive_cnt > 10:
        scores["dance"] *= 0.7

    # --- Правило для jumping ---
    total_frames = len(ntu_predictions)
    if total_frames > 0:
        cheer_up_percent = counter.get("cheer up", 0) / total_frames
        if cheer_up_percent > 0.80:
            scores["jump"] += 200.0
            scores["dance"] *= 0.1

    # --- Приоритет VLM ---
    if vlm_action:
        v_act = vlm_action.lower()
        if any(s in v_act for s in ["smoke", "smoking"]):
            scores["smoking_candidate"] += 1000 
        elif any(s in v_act for s in ["tug", "war", "rope"]):
            scores["tug_of_war"] = 1000
        elif any(s in v_act for s in ["rally", "meeting", "protest"]):
            scores["meeting"] = 1000
        elif any(s in v_act for s in ["circle", "triangle"]):
            scores["circle_triangle"] = 1000

    final_class = max(scores, key=scores.get)
    return final_class, scores


def map_ntu_to_target(ntu_class):
    """ Промежуточный маппинг """
    if ntu_class in FIGHT_CLASSES: return "fight"
    if ntu_class in HUG_CLASSES: return "hug"
    if ntu_class in HANDSHAKE_CLASSES: return "handshake"
    if ntu_class in DANCE_CLASSES: return "dance"
    if ntu_class in JUMP_CLASSES: return "jump"
    if ntu_class in WALK_CLASSES: return "walk"
    if ntu_class in SIT_CLASSES: return "sit"
    if ntu_class in STAND_CLASSES: return "stand"
    if ntu_class in SMOKING_CANDIDATE_CLASSES: return "smoking_candidate"
    return None