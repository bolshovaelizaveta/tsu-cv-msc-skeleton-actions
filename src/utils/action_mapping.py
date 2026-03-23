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

# Этот класс исключительно для vlm 
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
    """
    sequence: (T, V, C)
    """
    seq = np.array(sequence)

    if len(seq) < 2:
        return 0.0

    velocity = np.diff(seq[..., :2], axis=0)   # (T-1, V, 2)
    speed = np.linalg.norm(velocity, axis=-1)  # (T-1, V)

    return float(speed.mean())


# Добавлен аргумент vlm_action для связи с верификатором
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

    handshake_score = (
        counter.get("handshaking", 0) * 3.0 +
        counter.get("walking towards each other", 0) * 1.0
    )

    # dance делаем слабее, чем раньше
    dance_score = (
        counter.get("giving something to other person", 0) * 1.5 +
        counter.get("touch other person's pocket", 0) * 1.5 +
        counter.get("walking towards each other", 0) * 1.0 +
        counter.get("pat on back of other person", 0) * 1.0 +
        counter.get("hugging other person", 0) * 1.0 +
        counter.get("cheer up", 0) * 1.5
    )

    jump_score = (
        counter.get("jump up", 0) * 10.0 + 
        counter.get("hopping (one foot jumping)", 0) * 10.0 + 
        counter.get("cheer up", 0) * 4.0
    )

    walk_score = (
        counter.get("walking apart from each other", 0) * 4.0 +
        counter.get("walking towards each other", 0) * 4.0 +
        counter.get("staggering", 0) * 3.0 +
        counter.get("nausea/vomiting condition", 0) * 2.0
    )

    sit_score = (
        counter.get("sitting down", 0) * 5.0 +
        counter.get("typing on a keyboard", 0) * 3.0 +
        counter.get("reading", 0) * 2.0 +
        counter.get("writing", 0) * 2.0
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
        "smoking_candidate": smoking_candidate_score * 1.2 # Вес больше, так как руки у лица часто сбиваются
    }

    # --- motion-based correction ---
    if last_sequence is not None:
        motion = compute_motion_energy(last_sequence)

        # высокая энергия поддерживает fight / jump
        if motion > 0.08:
            scores["fight"] *= 1.35
            scores["jump"] *= 1.5
            scores["dance"] *= 0.90
            scores["walk"] *= 1.2
            scores["smoking_candidate"] *= 0.5 # При сильном движении курение маловероятно
        else:
            # плавное движение поддерживает dance / hug
            scores["dance"] *= 1.15
            scores["hug"] *= 1.05
            scores["sit"] *= 1.3
            scores["smoking_candidate"] *= 1.2 # Статичное положение увеличивает шанс курения/питья

    # --- rule-based overrides ---
    punch_cnt = counter.get("punching/slapping other person", 0)
    kick_cnt = counter.get("kicking other person", 0)
    push_cnt = counter.get("pushing other person", 0)

    aggressive_cnt = punch_cnt + kick_cnt + push_cnt

    # если есть заметное число явно агрессивных действий,
    # даем бонус fight
    if aggressive_cnt > 20:
        scores["fight"] += aggressive_cnt * 2.0

    # если fight близок к hug, но есть агрессивные классы — выбираем fight
    if aggressive_cnt > 10 and scores["fight"] > scores["hug"] * 0.7:
        scores["fight"] += 50

    # чтобы dance не доминировал на fight-сценах только из-за close-contact
    if aggressive_cnt > 10:
        scores["dance"] *= 0.7

    # правило для jumping
    total_frames = len(ntu_predictions)
    if total_frames > 0:
        jumping_total_percent = (counter.get("cheer up", 0) + counter.get("jump up", 0)) / total_frames
        if jumping_total_percent > 0.15: 
            scores["jump"] += 400.0  
            scores["dance"] *= 0.1

    # --- Приоритет VLM (Больше синонимов, чтобы точно поймать ответ) ---
    if vlm_action:
        v_act = str(vlm_action).lower()
        # КУРЕНИЕ
        if any(s in v_act for s in ["smoke", "smoking", "cigarette", "vape"]):
            scores["smoking_candidate"] += 5000 
        # КАНАТ
        elif any(s in v_act for s in ["tug", "war", "rope", "pulling"]):
            scores["tug_of_war"] = 5000
        # МИТИНГ
        elif any(s in v_act for s in ["rally", "meeting", "protest", "crowd", "gathering"]):
            scores["meeting"] = 5000
        # КРУГ / ХОРОВОД / ТАНЕЦ
        elif any(s in v_act for s in ["circle", "triangle", "formation", "dance", "round", "horovod"]):
            scores["circle_triangle"] = 5000

    final_class = max(scores, key=scores.get)

    return final_class, scores


# --- Функция для маппинга ---
def map_ntu_to_target(ntu_class):
    """
    Используется в infer_vlm.py для отрисовки действий на лету.
    """
    if ntu_class in JUMP_CLASSES: return "jump"
    if ntu_class in FIGHT_CLASSES: return "fight"
    if ntu_class in HUG_CLASSES: return "hug"
    if ntu_class in HANDSHAKE_CLASSES: return "handshake"
    if ntu_class in DANCE_CLASSES: return "dance"
    if ntu_class in WALK_CLASSES: return "walk"
    if ntu_class in SIT_CLASSES: return "sit"
    if ntu_class in STAND_CLASSES: return "stand"
    if ntu_class in SMOKING_CANDIDATE_CLASSES: return "smoking_candidate"
    
    return None