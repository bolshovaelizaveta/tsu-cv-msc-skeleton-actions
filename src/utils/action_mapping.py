NTU_TO_TARGET = {
    "hugging other person": "hug",
    "pat on back of other person": "hug",

    "handshaking": "handshake",
    "walking towards each other": "handshake",

    "punching/slapping other person": "fight",
    "kicking other person": "fight",
    "pushing other person": "fight",

    "walking apart from each other": "walk",
    "hopping (one foot jumping)": "jump",
    "jump up": "jump",
    "sitting down": "sit",
    "standing up": "stand",

    # close-contact classes, которые в киношных сценах часто возникают на танце
    "giving something to other person": "dance",
    "touch other person's pocket": "dance",
    "walking towards each other": "dance",
    "hugging other person": "dance",
    "pat on back of other person": "dance",
}


def map_ntu_to_target(ntu_class_name: str) -> str | None:
    return NTU_TO_TARGET.get(ntu_class_name, None)