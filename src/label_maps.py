LABEL_TO_ID_3CLASS = {
    "normal": 0,
    "offensive": 1,
    "hatespeech": 2,
}

ID_TO_LABEL_3CLASS = {v: k for k, v in LABEL_TO_ID_3CLASS.items()}

LABEL_TO_ID_BINARY = {
    "non_abusive": 0,
    "abusive": 1,
}

ID_TO_LABEL_BINARY = {v: k for k, v in LABEL_TO_ID_BINARY.items()}