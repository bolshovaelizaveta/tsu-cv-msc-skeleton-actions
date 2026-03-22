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

# Этот класс исключительно проверять через vlm 
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
    """
    sequence: (T, V, C)
    """
    seq = np.array(sequence)

    if len(seq) < 2:
        return 0.0

    velocity = np.diff(seq[..., :2], axis=0)   # (T-1, V, 2)
    speed = np.linalg.norm(velocity, axis=-1)  # (T-1, V)

    return float(speed.mean())


def resolve_target_class(ntu_predictions, last_sequence=None):
    counter = Counter(ntu_predictions)

    # --- базовые суммы по группам ---
    # fight делаем взвешенным
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
        "smoking_candidate": smoking_candidate_score * 1.2 # Вес больше, так как руки у лица часто сбиваются
    }

    # --- motion-based correction ---
    if last_sequence is not None:
        motion = compute_motion_energy(last_sequence)

        # высокая энергия поддерживает fight / jump
        if motion > 0.08:
            scores["fight"] *= 1.35
            scores["jump"] *= 1.15
            scores["dance"] *= 0.90
            scores["smoking_candidate"] *= 0.5 # При сильном движении курение маловероятно
        else:
            # плавное движение поддерживает dance / hug
            scores["dance"] *= 1.15
            scores["hug"] *= 1.05
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
        cheer_up_percent = counter.get("cheer up", 0) / total_frames
        
        # Если "cheer up" составляет более 80% всего времени окна
        if cheer_up_percent > 0.80:
            scores["jump"] += 200.0  # Гарантированно выводим прыжки в топ
            scores["dance"] *= 0.1   # Глушим танцы для этого конкретного окна

    final_class = max(scores, key=scores.get)

    return final_class, scores


def map_ntu_to_target(ntu_class):
    """
    Промежуточный маппинг для покадровой отрисовки (до вызова resolve_target_class).
    Возвращает целевой класс, если ntu_class найден в одной из групп.
    """
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