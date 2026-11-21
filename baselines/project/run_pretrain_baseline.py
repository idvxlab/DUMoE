#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""： ``--model NAME``  ``--models NAME`` 。"""

from __future__ import annotations

import sys
from pathlib import Path

CUR = Path(__file__).resolve().parent
ROOT = CUR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    argv = [a.strip() for a in sys.argv[1:] if a.strip()]
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            out.extend(["--models", argv[i + 1]])
            i += 2
        else:
            out.append(argv[i])
            i += 1
    sys.argv = [sys.argv[0]] + out
    from project.run_unified_pipeline import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
