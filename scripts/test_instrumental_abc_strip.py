#!/usr/bin/env python3
"""本地自测：纯伴奏 ABC 剥离 Vocal 声部（不依赖 GPU）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from handler import strip_vocal_voices_from_abc, wants_instrumental  # noqa: E402

SAMPLE = """X:1
T:demo
M:4/4
L:1/8
K:C
V:Vocal
C2 D2 E2 F2|G4 z4|
V:Ins
C,2 E,2 G,2 C2|E4 z4|
"""


def main() -> int:
    assert wants_instrumental("instrumental unspecified not applicable vocal folk")
    assert not wants_instrumental("pop female vocal")
    out, meta = strip_vocal_voices_from_abc(SAMPLE)
    assert meta["stripped"] is True, meta
    assert "V:Vocal" not in out
    assert "V:Ins" in out
    assert "C,2 E,2" in out
    assert "C2 D2 E2 F2" not in out
    print("ok", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
