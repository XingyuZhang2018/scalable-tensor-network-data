"""Redraw Figure 1(a), retaining the existing vector panels (b) and (c).

The assembled PDF is the source for the other panels; their original assembly
recipe is not in this repository. Repeated runs replace the same panel region.
Legacy background-grid strokes are disabled without changing other vector art.
"""
import importlib.util
import re
from io import BytesIO
from pathlib import Path

import fitz
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "order_plot", HERE / "plot_J2_results_stacked.py"
)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def remove_legacy_background_grids(document):
    """Remove only the two audited grid styles in the retained Figure 1 PDF.

    Earlier panel replacements left clipped copies of old panels in Form
    XObjects. Process each stream once, including those copies. Replace only
    the stroke operator of a single axis-aligned grid path with a non-painting
    end-path operator; text, clipping, fits and physical references stay intact.
    """
    number = rb"-?(?:\d+(?:\.\d*)?|\.\d+)"
    coordinates = rb"\s+(" + number + rb")\s+(" + number + rb")"
    styles = (("A3", 0.1, rb"0", rb"\.35"),
              ("A2-0", 0.3, rb"\.6901961", rb"\.8"))
    xrefs = set(document[0].get_contents())
    xrefs.update(item[0] for item in document[0].get_xobjects())
    removed = 0
    for xref in sorted(xrefs):
        stream = document.xref_stream(xref)
        if not stream:
            continue
        for state, opacity, gray, width in styles:
            kind, value = document.xref_get_key(
                xref, f"Resources/ExtGState/{state}/CA")
            if kind not in ("float", "int") or abs(float(value) - opacity) > 1e-6:
                continue
            pattern = re.compile(
                rb"(/" + state.encode() + rb"\s+gs\s+" + gray
                + rb"\s+G\s+2\s+J\s+1\s+j\s+" + width + rb"\s+w"
                + coordinates + rb"\s+m" + coordinates + rb"\s+l)\s+S\b")

            def end_grid_path(match):
                nonlocal removed
                x0, y0, x1, y1 = map(float, match.groups()[1:])
                if (x0 == x1 or y0 == y1) and max(abs(x1 - x0), abs(y1 - y0)) > 50:
                    removed += 1
                    return match.group(1) + b" n"
                return match.group(0)

            stream = pattern.sub(end_grid_path, stream)
        document.update_stream(xref, stream)
    return removed


def normalize_panel_b_label(document):
    """Update the retained vector panel's legacy magnetization label in place."""
    page = document[0]
    old = [ln for block in page.get_text("dict")["blocks"]
           for ln in block.get("lines", [])
           if "".join(s["text"] for s in ln["spans"]).replace(" ", "") == "M2(D,χ)"]
    if not old:
        return
    assert len(old) == 1
    label = old[0]
    old_rect = fitz.Rect(label["bbox"])
    with plt.rc_context({"mathtext.fontset": "stix", "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(20 / 72, 60 / 72))
        fig.text(0.5, 0.5, r"$m^2(D,\chi)$", rotation=90,
                 ha="center", va="center", fontsize=label["spans"][0]["size"])
        buffer = BytesIO()
        fig.savefig(buffer, format="pdf", transparent=True)
        plt.close(fig)
    overlay = fitz.open(stream=buffer.getvalue(), filetype="pdf")
    spans = [s for b in overlay[0].get_text("dict")["blocks"]
             for ln in b.get("lines", []) for s in ln["spans"]]
    dx = label["spans"][0]["origin"][0] - spans[0]["origin"][0]
    dy = (old_rect.y0 + old_rect.y1
          - min(s["bbox"][1] for s in spans) - max(s["bbox"][3] for s in spans)) / 2
    page.add_redact_annot(old_rect, fill=False, cross_out=False)
    page.apply_redactions(images=0, graphics=0, text=0)
    page.show_pdf_page(fitz.Rect(dx, dy, dx + 20, dy + 60), overlay, 0)


def remove_panel_b_j052_extrapolation(document):
    """Drop the retained J2=0.52 fit and its star, preserving all raw markers.

    The legacy panel uses viridis_r: J2=0.52 has RGB (.163625, .471133,
    .558148). Its fit is a 50-sample stroked path and its extrapolated star
    is a ten-vertex filled path. Raw markers are separate Form XObjects.
    Match those two audited styles only; repeated runs make no further edits.
    """
    number = rb"-?(?:\d+(?:\.\d*)?|\.\d+)"
    point = rb"\s+" + number + rb"\s+" + number
    color = rb"\.163625\s+\.471133\s+\.558148\s+RG\s+1\s+j\s+"
    patterns = (
        re.compile(rb"(" + color + rb"\.5\s+w" + point + rb"\s+m(?:"
                   + point + rb"\s+l){49})\s+S\b"),
        re.compile(rb"(" + color + rb"1\s+w" + point + rb"\s+m(?:"
                   + point + rb"\s+l){9}\s+h)\s+B\b"),
    )
    counts = [0, 0]
    xrefs = set(document[0].get_contents())
    xrefs.update(item[0] for item in document[0].get_xobjects())
    for xref in sorted(xrefs):
        original = document.xref_stream(xref)
        if not original:
            continue
        stream = original
        for i, pattern in enumerate(patterns):
            stream, count = pattern.subn(rb"\g<1> n", stream)
            counts[i] += count
        if stream != original:
            document.update_stream(xref, stream)
    if tuple(counts) not in ((0, 0), (1, 1)):
        raise ValueError(f"Unexpected J2=0.52 fit/star counts: {counts}")
    return tuple(counts)


def main():
    target = HERE / "J2_results_stacked.pdf"
    original = fitz.open(stream=target.read_bytes(), filetype="pdf")
    removed = remove_legacy_background_grids(original)
    print(f"Removed {removed} legacy background-grid strokes")
    fit_count, star_count = remove_panel_b_j052_extrapolation(original)
    print(f"Removed J2=0.52 extrapolation: {fit_count} fit, {star_count} star")
    normalize_panel_b_label(original)
    page_rect = original[0].rect
    metadata = original.metadata
    match = re.search(r"fig1-order-panel-width=([0-9.]+)", metadata.get("keywords", ""))
    previous_panel_width = float(match.group(1)) if match else 153.0
    panel_width = 175.0
    extra_width = panel_width - previous_panel_width
    source.INSET_LABEL_SIZE = 6.2
    source.NEEL_ARROW_LINEWIDTH = 1.05
    source.PLAQUETTE_LINEWIDTH = 1.1
    plt.rcParams.update({
        "font.size": 6.5, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
        "legend.fontsize": 5.7,
    })
    fig = plt.figure(figsize=(panel_width / 72, page_rect.height / 72))
    # Restore the original plotting width; reserve extra space between panels.
    ax = fig.add_axes([29 / panel_width, 0.19, 116 / panel_width, 0.73])
    bond = ax.twinx()
    fits = source.load_fit_results(HERE / "data/J2_sweep_fit_results.csv")
    bonds = source.load_vbs_order(HERE / "data/J2_sweep_vbs_order.csv")
    m_line, = ax.plot(
        [r["j2"] for r in fits], [r["M_ext"] for r in fits],
        color=source.NEEL_COLOR, marker="*", ms=2.8, lw=0.65,
        label=r"$m_{\mathrm{extr.}}=\sqrt{(m^2)_{\mathrm{extr.}}}$",
    )
    p_line, = bond.plot(
        [r["j2"] for r in bonds], [r["P"] for r in bonds],
        color=source.PLAQ_COLOR, marker="^", ms=2.3, lw=0.65,
        label=r"$(E_{\langle ij\rangle,\mathrm{max}}-E_{\langle ij\rangle,\mathrm{min}})/J_1$",
    )
    ax.axhline(0, color="0.35", lw=0.4, ls=":")
    ax.axvspan(0.485, 0.515, color="0.5", alpha=0.08, zorder=0)
    ax.set_xlim(0.39, 0.61)
    ax.set_xlabel(r"$J_2/J_1$", labelpad=2)
    ax.set_ylabel(m_line.get_label(), color=source.NEEL_COLOR, labelpad=0)
    bond.set_ylabel(p_line.get_label(), color=source.PLAQ_COLOR, labelpad=3)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.40, 0.45, 0.50, 0.55, 0.60]))
    for axis, color in ((ax, source.NEEL_COLOR), (bond, source.PLAQ_COLOR)):
        axis.set_ylim(-0.012, 0.19)
        axis.yaxis.set_major_locator(mticker.FixedLocator([0, 0.05, 0.10, 0.15]))
        axis.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        axis.tick_params(axis="y", colors=color, width=0.4, length=2, pad=1.5)
        for spine in axis.spines.values():
            spine.set_linewidth(0.5)
    ax.tick_params(axis="x", width=0.4, length=2, pad=2)
    ax.spines["left"].set_color(source.NEEL_COLOR)
    ax.spines["right"].set_visible(False)
    bond.spines["right"].set_color(source.PLAQ_COLOR)
    bond.spines["left"].set_visible(False)
    bond.spines["top"].set_visible(False)
    bond.spines["bottom"].set_visible(False)
    ax.grid(False, which="both", axis="both")
    bond.grid(False, which="both", axis="both")
    source.draw_neel_inset(ax)
    source.draw_plaquette_inset(ax)
    # The reusable inset helpers were authored for a larger source figure.
    for inset in ax.child_axes:
        for collection in inset.collections:
            collection.set_sizes(collection.get_sizes() * 0.65**2)
        for patch in inset.patches:
            if isinstance(patch, source.FancyArrowPatch):
                patch.set_mutation_scale(patch.get_mutation_scale() * 0.65)
    fig.text(15.7 / panel_width, 0.944, "a", fontsize=7.8, fontweight="bold")
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf")
    plt.close(fig)
    panel = fitz.open(stream=buffer.getvalue(), filetype="pdf")
    result = fitz.open()
    page = result.new_page(width=page_rect.width + extra_width, height=page_rect.height)
    tail = fitz.Rect(previous_panel_width, 0, page_rect.width, page_rect.height)
    destination = fitz.Rect(panel_width, 0, page.rect.width, page_rect.height)
    page.show_pdf_page(destination, original, 0, clip=tail)
    page.show_pdf_page(fitz.Rect(0, 0, panel_width, page_rect.height), panel, 0)
    original.close()
    keywords = re.sub(r"\s*fig1-order-panel-width=[0-9.]+", "", metadata.get("keywords", ""))
    metadata["keywords"] = f"{keywords} fig1-order-panel-width={panel_width}".strip()
    result.set_metadata(metadata)
    target.write_bytes(result.tobytes(garbage=4, deflate=True))
    result.close()
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
