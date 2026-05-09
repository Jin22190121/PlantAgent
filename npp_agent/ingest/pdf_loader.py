"""Stand-alone PDF text extractor for normal-operations procedure PDFs.

Uses zlib decompression on raw PDF streams (no pypdf dependency required) to
recover BT/ET text blocks. Good enough for the Westinghouse Section 19.0
training manual where text is rendered as `(...) Tj` operations.

Run as:
    python -m npp_agent.ingest.pdf_loader <path-to-pdf>
"""
from __future__ import annotations
import re
import sys
import zlib
from pathlib import Path
from typing import List


def extract_text(pdf_path: str | Path) -> str:
    raw = Path(pdf_path).read_bytes()
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.DOTALL)
    pieces: List[str] = []
    for s in streams:
        try:
            decompressed = zlib.decompress(s)
        except zlib.error:
            continue
        for block in re.findall(rb"BT(.*?)ET", decompressed, re.DOTALL):
            for txt in re.findall(rb"\(((?:[^()\\]|\\.)*)\)", block):
                try:
                    pieces.append(txt.decode("latin-1", errors="replace"))
                except Exception:
                    pass
    return " ".join(pieces)


def find_appendix_a_block(text: str) -> str:
    """Slice the Appendix 19-1 §A region (MODE 5 → 4) from the full text."""
    # Tolerant of soft hyphens / spaced letters as seen in extracted output.
    # Find the section A header
    start_re = re.compile(r"Heatup\s+from\s+COLD\s+SHUTDOWN\s+to\s+HOT\s+SHUTDOWN", re.I)
    end_re = re.compile(r"Heatup\s+from\s+HOT\s+SHUTDOWN\s+to\s+HOT\s+STANDBY", re.I)

    cleaned = re.sub(r"\s+", " ", text)
    m1 = start_re.search(cleaned)
    if not m1:
        return ""
    m2 = end_re.search(cleaned, m1.end())
    end = m2.start() if m2 else len(cleaned)
    return cleaned[m1.start():end]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    text = extract_text(argv[1])
    print(f"[ok] extracted {len(text)} chars")
    block = find_appendix_a_block(text)
    print(f"[ok] Appendix A block: {len(block)} chars")
    print("---")
    print(block[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
