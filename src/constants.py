# stores constants used throughout different py files

SHOT_LABELS = {
    "2ptShot",
    "3ptShot",
    "Layup",
    "FreeThrow",
    "PutBack",
    "Dunk",
}

# Normalization due to input file inconsistancy
LABEL_NORMALIZATION = {
    "Putback": "PutBack",
}

# Normalize only when the event itself is a shot.
SHOT_OUTCOME_MAP = {
    "made": "MADE",
    "missed": "MISSED",
    "outside": "MISSED",
    "fouled": "FOUL",
}