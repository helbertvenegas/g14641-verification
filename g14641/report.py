"""Minimal bookkeeping: each check records (statement, case, ok)."""

import sys
import time


class Report:
    def __init__(self, title):
        self.title = title
        self.rows = []
        self.t0 = time.time()
        print(f"== {title}")

    def check(self, statement, case, ok, detail=""):
        self.rows.append((statement, case, bool(ok), detail))
        if not ok:
            print(f"  FAIL  [{statement}] {case} {detail}")
        return ok

    def summary(self):
        by = {}
        for st, _, ok, _ in self.rows:
            a, b = by.get(st, (0, 0))
            by[st] = (a + ok, b + 1)
        for st, (a, b) in by.items():
            mark = "ok  " if a == b else "FAIL"
            print(f"  {mark} {st}: {a}/{b} cases")
        nfail = sum(1 for r in self.rows if not r[2])
        print(f"  -- {len(self.rows)} checks, {nfail} failures, {time.time() - self.t0:.1f}s")
        return nfail


def finish(rep):
    sys.exit(1 if rep.summary() else 0)
