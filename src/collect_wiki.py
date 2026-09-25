"""Build a small, fresh evaluation set from English Wikipedia talk pages.

Why: HateXplain is Twitter/Gab from 2019-2020. Scoring the models on recent
comments from a different platform shows how well they hold up on new data.

Ethics (details in docs/DATASHEET_wiki_talk.md):
- Uses the official MediaWiki API, never page scraping, following Wikimedia's
  API etiquette: descriptive User-Agent, one request at a time, `maxlag`,
  back-off on errors.
- Never requests or stores usernames or page titles (user-talk titles are
  usernames). Mentions, signatures, IPs and emails in the text become <user>.
- Content is CC BY-SA 4.0; every record keeps its revision id, which is both
  the attribution link and the handle for removal.
- Only revision ids and labels are committed. Text stays local and can be
  re-fetched with `rehydrate`.

Usage:
    python src/collect_wiki.py collect     # fetch talk-page comments
    python src/collect_wiki.py sample      # pick the items to label
    python src/collect_wiki.py rehydrate   # rebuild text from committed rev ids
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lxml import html as lxml_html

from utils import load_config

USER = "<user>"
_PH = " USERPLACEHOLDER "  # survives the HTML-tag stripping below

_PATTERNS = [
    # Signature timestamp: 12:34, 5 September 2026 (UTC)
    (re.compile(r"\d{1,2}:\d{2}, \d{1,2} \w+ \d{4} \(UTC\)"), " "),
    # Links to user pages / talk / contributions -> <user>
    (re.compile(r"\[\[\s*(?:User|User talk|Special:Contributions)\s*:[^\]]*\]\]", re.I), _PH),
    # Ping templates {{ping|Name}}, {{u|Name}}, {{reply to|Name}}
    (re.compile(r"\{\{\s*(?:ping|u|user|reply to|re|yo)\s*\|[^}]*\}\}", re.I), _PH),
    # Remaining templates
    (re.compile(r"\{\{[^{}]*\}\}"), " "),
    # [[target|shown text]] -> shown text ; [[target]] -> target
    (re.compile(r"\[\[[^\]|]*\|([^\]]*)\]\]"), r"\1"),
    (re.compile(r"\[\[([^\]]*)\]\]"), r"\1"),
    # External links [http://x shown] -> shown
    (re.compile(r"\[https?://\S+\s*([^\]]*)\]"), r"\1"),
    (re.compile(r"https?://\S+"), " "),
    # IP addresses and emails identify people too
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), _PH),
    (re.compile(r"\b[0-9a-f]{1,4}(?::[0-9a-f]{0,4}){4,7}\b", re.I), _PH),
    (re.compile(r"\S+@\S+\.\w+"), _PH),
    (re.compile(r"@\w+"), _PH),
    # HTML tags, bold/italic quotes, indentation markers
    (re.compile(r"<[^>]+>"), " "),
    (re.compile(r"'{2,}"), ""),
    (re.compile(r"(^|\n)[:*#]+"), r"\1"),
    (re.compile(r"\s+"), " "),
]


def clean_wikitext(text):
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    text = re.sub(r"\(\s*USERPLACEHOLDER\s*\)", " ", text)  # "(talk)" part of signatures
    text = re.sub(r"(?:USERPLACEHOLDER[\s,.;:]*)+", USER + " ", text)
    return re.sub(r"\s+", " ", text).strip(" -–—")


def parse_added_lines(diff_html):
    """Return lines that were purely added (new comments), not edits of existing lines."""
    if not diff_html or not diff_html.strip():
        return []
    root = lxml_html.fromstring(f"<table>{diff_html}</table>")
    added = []
    for row in root.xpath("//tr"):
        cells = row.xpath("./td")
        classes = " ".join(c.get("class", "") for c in cells)
        if "diff-addedline" in classes and "diff-deletedline" not in classes:
            for cell in row.xpath("./td[contains(@class, 'diff-addedline')]"):
                line = cell.text_content().strip()
                if line:
                    added.append(line)
    return added


class WikiClient:
    def __init__(self, config):
        self.url = config["api_url"]
        self.interval = config["request_interval_seconds"]
        self.maxlag = config["maxlag"]
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config["user_agent"]
        self._last = 0.0

    def get(self, params, retries=5):
        params = {**params, "format": "json", "formatversion": 2, "maxlag": self.maxlag}
        for attempt in range(retries):
            wait = self.interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            resp = self.session.get(self.url, params=params, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(int(resp.headers.get("Retry-After", 2 ** (attempt + 1))))
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("error", {}).get("code") == "maxlag":
                time.sleep(int(resp.headers.get("Retry-After", 5)))
                continue
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        raise RuntimeError("giving up after repeated errors / server lag")

    def recent_talk_edits(self, namespaces, limit):
        """Yield recent human edits on talk pages. Deliberately no 'user' or 'title' props."""
        params = {
            "action": "query",
            "list": "recentchanges",
            "rcnamespace": "|".join(map(str, namespaces)),
            "rctype": "edit",
            "rcshow": "!bot",
            "rcprop": "ids|timestamp|sizes",
            "rclimit": 500,
        }
        seen = 0
        while seen < limit:
            data = self.get(params)
            for rc in data["query"]["recentchanges"]:
                yield rc
                seen += 1
                if seen >= limit:
                    return
            if "continue" not in data:
                return
            params.update(data["continue"])

    def diff(self, old_revid, revid):
        data = self.get({"action": "compare", "fromrev": old_revid, "torev": revid, "prop": "diff"})
        return data.get("compare", {}).get("body", "")


def comment_from_diff(diff_html, min_chars, max_chars):
    text = clean_wikitext("\n".join(parse_added_lines(diff_html)))
    if len(text) < min_chars or len(text) > max_chars:
        return None
    if len(re.sub(r"<user>|\W", "", text)) < min_chars // 2:  # mostly markup / pings
        return None
    return text


def collect(config):
    out = Path(config["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["rev_id"] for line in out.open(encoding="utf-8")}
    client = WikiClient(config)
    kept = 0
    candidates = [
        rc
        for rc in client.recent_talk_edits(config["namespaces"], config["max_revisions"] * 3)
        if rc.get("old_revid") and rc["newlen"] - rc["oldlen"] >= config["min_chars"]
    ][: config["max_revisions"]]
    print(f"{len(candidates)} candidate edits; fetching diffs at 1 request/second")
    with out.open("a", encoding="utf-8") as f:
        for i, rc in enumerate(candidates, 1):
            if rc["revid"] in done:
                continue
            text = comment_from_diff(
                client.diff(rc["old_revid"], rc["revid"]), config["min_chars"], config["max_chars"]
            )
            if text:
                record = {
                    "rev_id": rc["revid"],
                    "old_rev_id": rc["old_revid"],
                    "namespace": rc["ns"],
                    "timestamp": rc["timestamp"],
                    "text": text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            if i % 100 == 0:
                print(f"  {i}/{len(candidates)} diffs, {kept} comments kept")
    print(f"done: {kept} new comments -> {out}")


def sample(config):
    """Choose items to label: half the baseline's most-flagged, half uniformly random.

    Pure random sampling would give very few abusive comments to evaluate on.
    Keeping the stratum lets the report show results for the random half on
    its own, which is the unbiased estimate.
    """
    from predict import load_predictor

    s = config["sample"]
    raw = pd.read_json(config["raw_path"], lines=True)
    raw["text_hash"] = raw["text"].map(lambda t: hashlib.sha1(t.encode()).hexdigest())
    raw = raw.drop_duplicates("text_hash").reset_index(drop=True)

    probs = load_predictor(s["scorer_config"]).predict_proba(raw["text"])
    raw["abusive_score"] = probs[:, 1] + probs[:, 2]

    rng = np.random.default_rng(s["seed"])
    n_flag = int(s["n"] * s["model_flagged_fraction"])
    flagged = raw.nlargest(n_flag, "abusive_score").assign(stratum="model_flagged")
    rest = raw.drop(flagged.index)
    n_rand = min(s["n"] - n_flag, len(rest))
    randoms = rest.iloc[rng.choice(len(rest), n_rand, replace=False)].assign(stratum="random")

    out = pd.concat([flagged, randoms]).sample(frac=1, random_state=s["seed"])
    cols = ["rev_id", "old_rev_id", "namespace", "timestamp", "stratum", "text"]
    out[cols].to_csv(config["to_label_path"], index=False)
    print(f"{len(out)} items ({n_flag} model-flagged, {n_rand} random) -> {config['to_label_path']}")


def rehydrate(config):
    """Rebuild the local text file from the committed label files (rev ids only)."""
    files = sorted(Path(config["labels_dir"]).glob("*.csv"))
    if not files:
        raise SystemExit(f"no label files in {config['labels_dir']}")
    ids = pd.concat(pd.read_csv(f) for f in files).drop_duplicates("rev_id")
    client = WikiClient(config)
    rows = []
    for rec in ids.itertuples():
        text = comment_from_diff(client.diff(rec.old_rev_id, rec.rev_id), 1, 10**6)
        if text:  # revisions deleted since collection are skipped, as they should be
            rows.append({"rev_id": rec.rev_id, "old_rev_id": rec.old_rev_id, "stratum": rec.stratum, "text": text})
    pd.DataFrame(rows).to_csv(config["to_label_path"], index=False)
    print(f"rehydrated {len(rows)}/{len(ids)} items -> {config['to_label_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect", "sample", "rehydrate"])
    parser.add_argument("--config", default="configs/collect_wiki.yaml")
    args = parser.parse_args()
    {"collect": collect, "sample": sample, "rehydrate": rehydrate}[args.command](load_config(args.config))
