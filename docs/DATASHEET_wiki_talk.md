# Datasheet: Wikipedia talk-page evaluation set

Follows the outline of *Datasheets for Datasets* (Gebru et al., 2021), shortened.

## Motivation
HateXplain comes from Twitter and Gab posts from 2019–2020. This set checks how models
trained on it hold up on **recent comments from a different platform**. It is an
**evaluation set only** and is never used for training.

## Collection
- **Source:** English Wikipedia article-talk (namespace 1) and user-talk (namespace 3) pages,
  from recent human edits (the `recentchanges` feed covers the last 30 days, bots excluded).
- **Method:** the official MediaWiki Action API (`list=recentchanges`, `action=compare`),
  never HTML scraping. Only lines *added* by an edit are kept, so each item is a new comment.
- **Access etiquette:** follows Wikimedia's API etiquette and User-Agent policy: a
  descriptive User-Agent with a project link, strictly serial requests at ≤ 1 per second,
  `maxlag=5`, and back-off on `429`, `5xx` and lag errors.
- **Scale:** about 1,500 diffs fetched and 500 comments labelled, a very light load.

## Privacy
- Usernames and page titles are **never requested** from the API. User-talk page titles are
  usernames, so they are excluded too.
- User links, pings, signatures, IP addresses, emails and @-mentions in the text are
  replaced with `<user>`, the same placeholder HateXplain uses.
- **What gets committed:** revision ids, sampling stratum and labels only
  (`data/external/labels/*.csv`). Comment text stays local (gitignored) and can be rebuilt
  with `python src/collect_wiki.py rehydrate`. A comment that Wikipedia deletes or
  suppresses disappears from rehydrated copies automatically.
- **Removal requests:** open an issue with the revision id and it will be removed from the
  label files.

## Licence
Wikipedia text is licensed under **CC BY-SA 4.0**. Attribution for each item is its
revision: `https://en.wikipedia.org/w/index.php?diff=<rev_id>`. Any redistributed text must
keep that licence.

## Sampling and labelling
- 50% **model-flagged**: the comments the TF-IDF baseline scores as most abusive.
  Without this, a random sample of talk pages contains too few abusive comments to measure.
- 50% **uniform random** from the rest.
- Results are reported per stratum. The random stratum is the unbiased estimate; the
  flagged stratum mostly measures precision on what the baseline already suspects.
- Labelled by the project author using `docs/LABELING_GUIDELINES.md`, with a second
  annotator on 150 items (Cohen's kappa reported in `reports/RESULTS.md`).

## Known limitations
- One month of English Wikipedia only. Wikipedia moderates heavily and has strong civility
  norms, so identity-based hate is rarer than on Gab.
- Model-flagged sampling favours abuse that looks like HateXplain, so it can understate
  how much novel abuse the models miss. The random stratum partly corrects for this.
- Labelled by one to two people, not a diverse annotator pool.
