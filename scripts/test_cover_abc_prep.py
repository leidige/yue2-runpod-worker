"""Local unit checks for Cover ABC prep (no GPU)."""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# handler imports runpod at module level; stub for local unit checks
sys.modules.setdefault(
    "runpod",
    types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda *_a, **_k: None)),
)
sys.modules.setdefault("requests", types.ModuleType("requests"))

from handler import (  # noqa: E402
    prefer_vocal_melody_abc,
    prepare_cover_abc,
    strip_chord_symbols_from_abc,
)


SAMPLE = """X:1
T:test
M:4/4
L:1/8
K:C
V:Vocal
"C" C2 D2 E2 F2 | "G" G2 A2 B2 c2 |
V:Ins
E2 F2 G2 A2 | B2 c2 d2 e2 |
"""


def main() -> None:
    cleaned, meta = strip_chord_symbols_from_abc(SAMPLE)
    assert meta["chord_quotes_removed"] == 2, meta
    assert '"C"' not in cleaned and '"G"' not in cleaned

    pref, pmeta = prefer_vocal_melody_abc(SAMPLE)
    assert pmeta.get("preferred") is True, pmeta
    assert "V:Ins" not in pref
    assert "V:Vocal" in pref
    assert "C2 D2" in pref

    out, info = prepare_cover_abc(SAMPLE, cot="melody", instrumental=False)
    assert info["chords"]["chord_quotes_removed"] == 2
    assert info.get("voices") == "keep_vocal_and_ins"
    assert out and "V:Ins" in out and "V:Vocal" in out
    assert '"C"' not in out

    out_full, info_full = prepare_cover_abc(SAMPLE, cot="full", instrumental=False)
    assert info_full["chords"].get("kept_for_full") is True
    assert '"C"' in out_full and '"G"' in out_full

    out2, info2 = prepare_cover_abc(
        SAMPLE, cot="melody", instrumental=False, prefer_vocal_only=True
    )
    assert info2["vocal_prefer"]["preferred"] is True
    assert out2 and "V:Ins" not in out2
    print("OK", info, info_full, info2)


if __name__ == "__main__":
    main()
