from collect_wiki import WikiClient, clean_wikitext, comment_from_diff, parse_added_lines

DIFF = (
    '<tr><td colspan="2" class="diff-empty diff-side-deleted"></td><td class="diff-marker">+</td>'
    '<td class="diff-addedline diff-side-added">'
    "<div>This is a brand new comment on the talk page.</div></td></tr>"
    '<tr><td class="diff-marker">-</td><td class="diff-deletedline"><div>old line</div></td>'
    '<td class="diff-marker">+</td><td class="diff-addedline"><div>old line, edited</div></td></tr>'
)


def test_parse_added_lines_keeps_only_new_lines():
    assert parse_added_lines(DIFF) == ["This is a brand new comment on the talk page."]
    assert parse_added_lines("") == []


def test_clean_wikitext_removes_identities():
    raw = (
        ":::You are wrong {{ping|SomeEditor}}, see [[Talk:Foo#Bar|this thread]] https://example.com "
        "[[User:Abc|Abc]] ([[User talk:Abc|talk]]) 12:34, 5 September 2026 (UTC)"
    )
    out = clean_wikitext(raw)
    assert "SomeEditor" not in out and "Abc" not in out and "UTC" not in out
    assert "example.com" not in out
    assert out.startswith("You are wrong <user>")
    assert "this thread" in out
    assert "<user>" in clean_wikitext("Reverted by 192.168.0.1 and mail a@b.com")
    assert "192.168" not in clean_wikitext("Reverted by 192.168.0.1")


def test_comment_length_filter():
    assert comment_from_diff(DIFF, 20, 500) == "This is a brand new comment on the talk page."
    assert comment_from_diff(DIFF, 200, 500) is None


class FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, pages):
        self.pages, self.calls, self.headers = pages, [], {}

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        return FakeResponse(self.pages[len(self.calls) - 1])


def test_recent_changes_pages_and_never_requests_usernames():
    config = {"api_url": "http://x", "request_interval_seconds": 0, "maxlag": 5, "user_agent": "test"}
    client = WikiClient(config)
    rc = [
        {"revid": i, "old_revid": i - 1, "ns": 1, "timestamp": "t", "oldlen": 0, "newlen": 99}
        for i in range(3)
    ]
    client.session = FakeSession(
        [
            {"query": {"recentchanges": rc[:2]}, "continue": {"rccontinue": "abc"}},
            {"query": {"recentchanges": rc[2:]}},
        ]
    )
    got = list(client.recent_talk_edits([1, 3], limit=10))
    assert [r["revid"] for r in got] == [0, 1, 2]
    first, second = client.session.calls
    assert "user" not in first["rcprop"] and "title" not in first["rcprop"]
    assert second["rccontinue"] == "abc" and first["maxlag"] == 5
