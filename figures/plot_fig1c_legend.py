"""Enlarge the assembled Figure 1(c) legend without redrawing any data.

The original legend occupies a data-free lower-left region of panel (c).
Only its text and fully contained marker/line paths are removed; grid lines
and the original vector panels are preserved. The legend remains inside
panel (c), clear of every data marker, with the original page dimensions.
"""
import argparse
from io import BytesIO
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "J2_results_stacked.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "J2_results_stacked.pdf")
    args = parser.parse_args()
    doc = fitz.open(stream=args.source.read_bytes(), filetype="pdf")
    page = doc[0]
    metadata = doc.metadata
    keywords = metadata.get("keywords", "")
    fitz.TOOLS.set_small_glyph_heights(True)
    updated = "fig1-c-legend=inside" in keywords
    if updated:
        print("Enlarged inline legend is already present; no changes made.")
        return
    # The nested original Form retains its rounded frame through redaction.
    # Discard just that known frame path's paint operator, preserving graphics state.
    frame_start = b".8 G 1 w 75.07 47.154767 m"
    removed_frames = 0
    for xref in range(1, doc.xref_length()):
        if not doc.xref_is_stream(xref):
            continue
        content = doc.xref_stream(xref)
        start = content.find(frame_start)
        if start >= 0:
            end = content.index(b" h B", start) + 3
            doc.update_stream(xref, content[:end] + b"n" + content[end + 1:])
            removed_frames += 1
    assert removed_frames == 1, "Expected one original legend frame"
    # Remove the original legend frame and handles, retaining longer grid paths.
    # Include the stroked frame's miter extent, but stop above the bottom ticks.
    page.add_redact_annot(fitz.Rect(400, 94, 478, 134),
                         fill=False, cross_out=False)
    page.apply_redactions(images=0, graphics=1, text=1)
    erase = fitz.Rect(405.9, 99.3, 471.5, 128.5)
    page.add_redact_annot(erase, fill=False, cross_out=False)
    page.apply_redactions(images=0, graphics=1, text=0)
    # Relocate the coupling annotation above the enlarged legend.
    if not updated:
        labels = [fitz.Rect(ln["bbox"]) for block in page.get_text("dict")["blocks"]
                  for ln in block.get("lines", [])
                  if ln["bbox"][0] > 390
                  and "".join(s["text"] for s in ln["spans"]).replace(" ", "") == "J2/J1=0.50"]
        assert len(labels) == 1
        page.add_redact_annot(labels[0], fill=False, cross_out=False)
        page.apply_redactions(images=0, graphics=0, text=0)
    assert "this work" not in page.get_text()
    width, height = page.rect.width, page.rect.height
    handles = [
        Line2D([], [], color="C0", marker="o", linestyle="none", markersize=3.73, label="this work"),
        Line2D([], [], color="navy", linestyle="--", linewidth=0.9, label="iPEPS fit"),
        Line2D([], [], color="C2", linestyle="-.", linewidth=0.9, label=r"$C_{4v}$ iPEPS"),
        Line2D([], [], color="black", marker="s", linestyle="none", markersize=3.73, label="CNN-MPS"),
        Line2D([], [], color="C5", marker="P", markeredgecolor="black", markeredgewidth=0.4,
               linestyle="none", markersize=5.26, label="Res.+Lanc."),
        Line2D([], [], color="C3", marker="^", markeredgecolor="black", markeredgewidth=0.4,
               linestyle="none", markersize=5.26, label="ViT"),
        Line2D([], [], color="darkviolet", marker="D", markeredgecolor="black", markeredgewidth=0.4,
               linestyle="none", markersize=4.58, label="PBC PEPS"),
        Line2D([], [], color="darkorange", marker="v", markeredgecolor="black", markeredgewidth=0.4,
               linestyle="none", markersize=5.26, label="OBC PEPS"),
    ]
    with plt.rc_context({"font.family": "Arial", "mathtext.fontset": "stix", "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(width / 72, height / 72))
        fig.text(405 / width, 1 - 95 / height, r"$J_2/J_1=0.50$",
                 fontsize=5.5118108, ha="left", va="baseline")
        legend = fig.legend(handles=handles, loc="upper left",
                            bbox_to_anchor=(404 / width, 1 - 99 / height),
                            ncol=2, fontsize=5.5118108, frameon=True, fancybox=False,
                            facecolor="white", edgecolor="0.8", framealpha=1,
                            handlelength=1.25, handletextpad=0.4, columnspacing=0.85,
                            labelspacing=0.30, borderpad=0.35, borderaxespad=0)
        legend.get_frame().set_linewidth(0.55)
        fig.canvas.draw()
        bounds = legend.get_window_extent().transformed(fig.transFigure.inverted())
        assert 0 <= bounds.x0 < bounds.x1 <= 1 and 0 <= bounds.y0 < bounds.y1 <= 1, bounds
        buffer = BytesIO()
        fig.savefig(buffer, format="pdf", transparent=True)
        plt.close(fig)
    overlay = fitz.open(stream=buffer.getvalue(), filetype="pdf")
    page.show_pdf_page(page.rect, overlay, 0)
    if not updated:
        metadata["keywords"] = f"{keywords} fig1-c-legend=inside".strip()
    doc.set_metadata(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(doc.tobytes(garbage=4, deflate=True))
    print(f"Saved {args.output}; legend fontsize=5.5118108 PDF pt")


if __name__ == "__main__":
    main()
