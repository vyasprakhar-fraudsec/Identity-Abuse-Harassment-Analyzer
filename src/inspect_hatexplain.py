from datasets import load_dataset
from collections import Counter
import json

ds = load_dataset(
    "Hate-speech-CNERG/hatexplain",
    trust_remote_code=True
)

print("=== DATASET SHAPE ===")
for split in ds:
    print(f"{split}: {len(ds[split])}")

print("\n=== SAMPLE EXAMPLE ===")
sample = ds["train"][0]
print(json.dumps(sample, indent=2)[:2000])

label_counter = Counter()
target_counter = Counter()

for i in range(min(1000, len(ds["train"]))):
    example = ds["train"][i]
    ann = example["annotators"]

    labels = ann["label"]          # list of ints
    targets = ann["target"]        # list of list[str]

    for j, lbl in enumerate(labels):
        label_counter[lbl] += 1

        for tgt in targets[j]:
            if tgt:
                target_counter[tgt] += 1

print("\n=== LABEL COUNTS (annotator-level, first 1000 train examples) ===")
print(label_counter)

print("\n=== TOP TARGET GROUPS (annotator-level, first 1000 train examples) ===")
for target, count in target_counter.most_common(20):
    print(f"{target}: {count}")
    
for i in range(5):
    ex = ds["train"][i]
    text = " ".join(ex["post_tokens"])
    print("TEXT:", text)
    print("LABELS:", ex["annotators"]["label"])
    print("TARGET:", ex["annotators"]["target"])
    print("---")