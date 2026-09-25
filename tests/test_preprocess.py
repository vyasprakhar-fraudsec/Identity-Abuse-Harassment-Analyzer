from preprocess import build_frames, majority_label, majority_targets


def ann(label, targets):
    return {"label": label, "target": targets}


def test_majority_label_needs_two_votes():
    assert majority_label([ann("normal", []), ann("normal", []), ann("offensive", [])]) == "normal"
    assert majority_label([ann("normal", []), ann("offensive", []), ann("hatespeech", [])]) is None


def test_majority_targets_needs_two_annotators_and_drops_none():
    annotators = [
        ann("hatespeech", ["African", "Women"]),
        ann("hatespeech", ["African"]),
        ann("offensive", ["None"]),
    ]
    assert majority_targets(annotators) == ["African"]
    assert majority_targets([ann("normal", ["None"])] * 3) == []


def test_build_frames_uses_official_split_and_drops_undecided():
    dataset = {
        "p1": {"annotators": [ann("hatespeech", ["Islam"])] * 3, "post_tokens": "they are all bad".split()},
        "p2": {
            "annotators": [ann("normal", []), ann("offensive", []), ann("hatespeech", [])],
            "post_tokens": "undecided post here".split(),
        },
        "p3": {
            "annotators": [ann("normal", ["None"])] * 3,
            "post_tokens": "nice day today http://x.co".split(),
        },
    }
    divisions = {"train": ["p1", "p2"], "test": ["p3"]}
    config = {
        "preprocessing": {
            "lowercase": False,
            "remove_urls": True,
            "remove_extra_whitespace": True,
            "min_tokens": 3,
        }
    }
    frames, stats = build_frames(dataset, divisions, config)
    assert list(frames["train"]["post_id"]) == ["p1"]
    assert frames["train"].iloc[0]["target_groups"] == "Islam"
    assert frames["test"].iloc[0]["text"] == "nice day today"
    assert stats["dropped_undecided"] == 1
