"""
generate_pdf.py
===============
Generates publication/final_paper.pdf from publication/final_paper.md.

Pipeline:
  1. Pre-process the markdown — inserts actual ![](path) image blocks at
     the correct positions after each figure reference sentence.
  2. pandoc  →  standalone HTML  (handles all markdown formatting)
  3. Post-process HTML — injects academic CSS, tightens figure styling.
  4. weasyprint  →  PDF  (renders HTML with embedded images at 300 DPI).

Usage:
    python publication/generate_pdf.py

Requirements:
    pandoc  (brew install pandoc)
    weasyprint  (pip install weasyprint)
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
PUB_DIR = Path(__file__).parent          # publication/
FIG_DIR = PUB_DIR / "figures"
MD_SRC  = PUB_DIR / "final_paper.md"
OUT_PDF = PUB_DIR / "final_paper.pdf"


# ── Dependency check ─────────────────────────────────────────────────────────
def check_deps():
    if not shutil.which("pandoc"):
        sys.exit("ERROR: pandoc not found. Install with: brew install pandoc")
    try:
        import weasyprint  # noqa: F401
    except ImportError:
        sys.exit("ERROR: weasyprint not found. Install with: pip install weasyprint")
    for fig in ["fig1_system_architecture.png", "fig2_accuracy_comparison.png",
                "fig3_confusion_matrices.png", "fig4_processing_time.png"]:
        if not (FIG_DIR / fig).exists():
            sys.exit(f"ERROR: Missing figure file: {FIG_DIR / fig}\n"
                     "Run: python publication/generate_figures.py")


# ── Step 1: Pre-process markdown ─────────────────────────────────────────────
# These are the exact sentences that end each figure-reference paragraph.
# We append a proper Markdown image block immediately after each one so that
# pandoc sees real image syntax and embeds the figure.

FIGURE_INJECTIONS = [
    # (sentence to match,  figure filename,  caption text)
    (
        "Figure 1 illustrates the full system pipeline.",
        "fig1_system_architecture.png",
        "**Figure 1.** MalariaLLM System Architecture. "
        "Five-stage pipeline from Patient Input through the Rule-Based "
        "Classification Engine, LLM Reasoning Module, Structured Clinical "
        "Output, and PDF Report. The LLM receives the rule-derived severity "
        "classification and cannot alter it.",
    ),
    (
        "Figure 2 visualizes accuracy across all stages for both systems.",
        "fig2_accuracy_comparison.png",
        "**Figure 2.** Diagnostic Accuracy: Baseline vs. LLM-Enhanced Hybrid "
        "System. Side-by-side bars for No Malaria, Stage I, Stage II, Critical, "
        "and Overall (n = 1,682). pp = percentage points.",
    ),
    (
        "Row-normalised confusion matrices for both systems are presented in Figure 3.",
        "fig3_confusion_matrices.png",
        "**Figure 3.** Confusion Matrices: Row-Normalised per True Stage "
        "(n = 1,682). *Left:* Baseline (Rule-Based Only). "
        "*Right:* LLM-Enhanced Hybrid. Cell values show row-normalised "
        "percentage and raw case count.",
    ),
    (
        "Figure 4 presents the processing time distribution and the "
        "accuracy\u2013latency trade-off.",
        "fig4_processing_time.png",
        "**Figure 4.** Processing Efficiency. *Left:* LLM inference time "
        "distribution with mean (6.3 s) and median (5.4 s) reference lines. "
        "*Right:* Accuracy\u2013latency trade-off: baseline (<0.01 s, 30.62%) "
        "vs. hybrid (6.3 s, 87.22%).",
    ),
]


def preprocess_markdown(text: str) -> str:
    """Insert ![caption](path) blocks after each figure-reference sentence."""
    for (sentence, fname, caption) in FIGURE_INJECTIONS:
        # Relative path from publication/ so weasyprint base_url works
        rel_path = f"figures/{fname}"
        img_block = f"\n\n![{caption}]({rel_path})\n\n"
        if sentence in text:
            text = text.replace(sentence, sentence + img_block, 1)
        else:
            print(f"  WARNING: Could not find figure injection anchor:\n"
                  f"    «{sentence[:70]}…»")

    # Clean up the List of Figures path references (replace backtick paths
    # with actual embedded images so they also render in the PDF).
    lof_pattern = re.compile(
        r'(`publication/figures/(fig\d_\w+\.png)`)',
    )
    def replace_lof(m):
        fname = m.group(2)
        return f"![](figures/{fname})"

    text = lof_pattern.sub(replace_lof, text)
    return text


# ── Step 2: pandoc markdown → HTML ───────────────────────────────────────────

PANDOC_CSS = """\
/* Academic paper CSS — weasyprint-compatible */
@page {
  size: A4;
  margin: 2.2cm 2.5cm 2.5cm 2.5cm;
  @bottom-center {
    content: counter(page);
    font-size: 9pt;
    color: #64748b;
  }
}

body {
  font-family: "Linux Libertine", "Georgia", "Times New Roman", serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #0f172a;
  text-align: justify;
  hyphens: auto;
}

h1 {
  font-size: 15pt;
  font-weight: bold;
  text-align: center;
  margin-top: 0;
  margin-bottom: 0.4em;
  color: #0f172a;
  line-height: 1.25;
}

/* Running title / author lines */
p strong:only-child {
  display: block;
}

h2 {
  font-size: 12pt;
  font-weight: bold;
  margin-top: 1.6em;
  margin-bottom: 0.3em;
  color: #0f172a;
  border-bottom: 0.5pt solid #cbd5e1;
  padding-bottom: 2pt;
  page-break-after: avoid;
}

h3 {
  font-size: 11pt;
  font-weight: bold;
  margin-top: 1.2em;
  margin-bottom: 0.2em;
  color: #0f172a;
  page-break-after: avoid;
}

h4 {
  font-size: 10.5pt;
  font-weight: bold;
  font-style: italic;
  margin-top: 1em;
  margin-bottom: 0.15em;
  color: #334155;
  page-break-after: avoid;
}

p {
  margin: 0.4em 0 0.6em 0;
  orphans: 3;
  widows: 3;
}

/* Abstract box */
div.abstract {
  margin: 1em 1.5em;
  padding: 0.8em 1em;
  border-left: 3pt solid #0284c7;
  background: #f0f9ff;
  font-size: 10pt;
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 9.5pt;
  margin: 0.8em 0 1.2em 0;
  page-break-inside: avoid;
}

thead tr {
  background: #1e293b;
  color: white;
}

th {
  padding: 5pt 7pt;
  text-align: left;
  font-weight: bold;
}

td {
  padding: 4pt 7pt;
  border-bottom: 0.4pt solid #e2e8f0;
}

tr:nth-child(even) td {
  background: #f8fafc;
}

caption {
  font-size: 9.5pt;
  font-weight: bold;
  text-align: left;
  margin-bottom: 4pt;
  color: #334155;
}

/* Figures */
figure {
  margin: 1em 0 1.2em 0;
  text-align: center;
  page-break-inside: avoid;
}

figure img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}

figcaption {
  font-size: 9pt;
  color: #475569;
  margin-top: 5pt;
  text-align: left;
  font-style: italic;
  line-height: 1.35;
}

/* Inline images (no figure wrapper) */
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.6em auto 0.2em auto;
}

/* Blockquotes / code */
code {
  font-family: "Courier New", monospace;
  font-size: 9pt;
  background: #f1f5f9;
  padding: 1pt 3pt;
  border-radius: 2pt;
}

pre {
  background: #f8fafc;
  border: 0.5pt solid #e2e8f0;
  border-left: 3pt solid #0284c7;
  padding: 8pt 10pt;
  font-size: 8.5pt;
  overflow: hidden;
  white-space: pre-wrap;
  word-break: break-all;
  margin: 0.8em 0;
  page-break-inside: avoid;
}

/* Horizontal rule */
hr {
  border: none;
  border-top: 0.5pt solid #cbd5e1;
  margin: 1.2em 0;
}

/* Lists */
ul, ol {
  margin: 0.3em 0 0.6em 0;
  padding-left: 1.5em;
}

li {
  margin-bottom: 0.2em;
}

/* References */
ol > li:target {
  background: #fef9c3;
}

/* Strong / em */
strong { color: #0f172a; }
em { font-style: italic; }

/* Prevent page breaks inside table rows */
tr { page-break-inside: avoid; }
"""


def run_pandoc(md_path: Path, html_path: Path):
    """Convert markdown → standalone HTML via pandoc."""
    cmd = [
        "pandoc", str(md_path),
        "--standalone",
        "--from", "markdown+smart",
        "--to", "html5",
        "--output", str(html_path),
        "--metadata", "lang=en",
        "--wrap=none",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  pandoc stderr:\n{result.stderr}")
        sys.exit(f"ERROR: pandoc failed (exit {result.returncode})")


# ── Step 3: Post-process HTML ─────────────────────────────────────────────────

def postprocess_html(html: str, css: str, base_url: str) -> str:
    """
    - Inject the academic CSS.
    - Wrap standalone <img> tags in <figure> + <figcaption> blocks so that
      the alt text (our caption) is rendered below the image.
    - Tighten the title block layout.
    """
    # Inject CSS into <head>
    style_tag = f"\n<style>\n{css}\n</style>\n"
    html = html.replace("</head>", style_tag + "</head>", 1)

    # Wrap <img> blocks in <figure><figcaption> so captions render correctly.
    # pandoc renders ![caption](path) as <img alt="caption" src="path"> inside <p>.
    def wrap_img(m):
        tag   = m.group(0)
        alt   = re.search(r'alt="([^"]*)"', tag)
        src   = re.search(r'src="([^"]*)"', tag)
        if not alt or not src:
            return tag
        alt_text = alt.group(1)
        src_val  = src.group(1)
        return (
            f'<figure>\n'
            f'  <img src="{src_val}" alt="" '
            f'style="max-width:100%;display:block;margin:0 auto;">\n'
            f'  <figcaption>{alt_text}</figcaption>\n'
            f'</figure>'
        )

    # Replace <p><img ...></p> patterns
    html = re.sub(
        r'<p>\s*(<img[^>]+>)\s*</p>',
        lambda m: wrap_img(m.group(1)),
        html,
        flags=re.DOTALL,
    )

    return html


# ── Step 4: weasyprint HTML → PDF ─────────────────────────────────────────────

def render_pdf(html_path: Path, pdf_path: Path, base_url: str):
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration

    font_config = FontConfiguration()
    doc = HTML(filename=str(html_path), base_url=base_url)
    doc.write_pdf(str(pdf_path), font_config=font_config)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\nMalariaLLM — PDF Generator")
    print("===========================\n")

    check_deps()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tmp_md   = tmp / "paper.md"
        tmp_html = tmp / "paper.html"
        tmp_css  = tmp / "style.css"

        # ── Step 1: Pre-process markdown ─────────────────────────────────────
        print("  [1/4] Pre-processing markdown …")
        raw_md = MD_SRC.read_text(encoding="utf-8")
        processed = preprocess_markdown(raw_md)
        tmp_md.write_text(processed, encoding="utf-8")

        # Also write the CSS file (pandoc --css reads it; weasyprint inlines it)
        tmp_css.write_text(PANDOC_CSS, encoding="utf-8")

        # ── Step 2: pandoc → HTML ─────────────────────────────────────────────
        print("  [2/4] Running pandoc (markdown → HTML) …")
        run_pandoc(tmp_md, tmp_html)

        # ── Step 3: Post-process HTML ─────────────────────────────────────────
        print("  [3/4] Post-processing HTML (CSS + figure wrapping) …")
        html_text = tmp_html.read_text(encoding="utf-8")
        html_text = postprocess_html(html_text, PANDOC_CSS, base_url=str(PUB_DIR))
        tmp_html.write_text(html_text, encoding="utf-8")

        # ── Step 4: weasyprint → PDF ──────────────────────────────────────────
        print("  [4/4] Rendering PDF via weasyprint …")
        # base_url must point to publication/ so that relative paths like
        # "figures/fig1.png" resolve correctly from the temp HTML file.
        render_pdf(tmp_html, OUT_PDF, base_url=str(PUB_DIR) + "/")

    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"\n  PDF saved: {OUT_PDF}")
    print(f"  Size     : {size_kb:.0f} KB\n")
    print("  Done.\n")


if __name__ == "__main__":
    main()
