"""--rw-shard K/N, Playwright's --shard: the suite is split over N machines, each test in exactly one part."""

import re

TEST = "\n".join(f"def test_{i}():\n    pass\n" for i in range(24))


def _ran(r) -> set[str]:
    return set(re.findall(r"::(test_\d+) PASSED", r.stdout.str()))


def test_the_parts_cover_the_suite_once(pytester):
    pytester.makepyfile(test_many=TEST)
    parts = [_ran(pytester.runpytest("-p", "no:cacheprovider", "-v", "--rw-shard", f"{k}/3")) for k in (1, 2, 3)]
    assert set().union(*parts) == {f"test_{i}" for i in range(24)}
    assert sum(len(p) for p in parts) == 24  # no test in two parts
    assert all(parts)  # and none of the three empty
    again = _ran(pytester.runpytest("-p", "no:cacheprovider", "-v", "--rw-shard", "2/3"))
    assert again == parts[1]  # the same split every time


def test_a_bad_shard_is_a_usage_error(pytester):
    pytester.makepyfile(test_many=TEST)
    for bad in ("0/3", "4/3", "two"):
        r = pytester.runpytest("-p", "no:cacheprovider", "--rw-shard", bad)
        assert r.ret == 4, bad  # pytest's usage error
        assert "--rw-shard" in r.stderr.str()
