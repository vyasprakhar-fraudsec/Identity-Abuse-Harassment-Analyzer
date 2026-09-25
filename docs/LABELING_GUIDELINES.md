# Labelling guidelines: Wikipedia talk-page comments

The labels match HateXplain's so the models can be scored directly. When unsure,
choose the **less severe** label, and use `u` (unclear) when the comment can't be
judged without context you don't have.

| Key | Label | Use when | Examples of the pattern |
|---|---|---|---|
| `3` | **hatespeech** | Attacks, dehumanises or calls for harm against people **because of a protected identity** (race, religion, ethnicity, nationality, gender, sexual orientation, disability). | Slurs aimed at a group; "those people are vermin"; saying a group should be removed or harmed. |
| `2` | **offensive** | Insulting, abusive or profane, but **not** aimed at an identity group. Personal attacks on an editor go here. | "You're an idiot"; swearing at someone; mocking a user's competence. |
| `1` | **normal** | Everything else, including heated but civil disagreement. | "This source is unreliable and your edit is wrong"; blunt policy arguments. |
| `u` | unclear | Not English, only markup, or impossible to judge without the thread. | Excluded from evaluation. |

## Edge cases
- **Quoting or reporting abuse** (e.g. "User X called me a slur, please act") → `normal`. The author isn't being abusive.
- **Discussing an identity topic** neutrally, even a sensitive one → `normal`.
- **Reclaimed or in-group language** used without hostility → `normal`.
- **Insult with a passing identity mention** → `hatespeech` only if the identity is the *reason* for the attack; otherwise `offensive`.
- **Sarcasm**: label what the comment actually does to its target, not its surface words.

## Agreement check
A second person labels the first 150 items without seeing your labels
(`python src/label_wiki.py --annotator <name> --limit 150`). `evaluate_ood.py`
reports Cohen's kappa. Around 0.4–0.6 is typical for this task; HateXplain reports
Krippendorff's alpha of 0.46 for its annotators.
