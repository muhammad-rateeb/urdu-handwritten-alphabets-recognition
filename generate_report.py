import json
from pathlib import Path

from fpdf import FPDF, XPos, YPos


def load_summary(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report_md(summary, output_path: Path):
    lines = []
    lines.append("# Handwritten Urdu Alphabets Recognition System")
    lines.append("")
    lines.append("## 1. Problem Statement")
    lines.append(
        "Build a system to recognize handwritten Urdu alphabets from mobile-camera images collected by the student."
    )
    lines.append("")
    lines.append("## 2. Dataset")
    lines.append(f"- Total classes: {len(summary['classes'])}")
    lines.append(f"- Images per class: 72")
    lines.append(f"- Image size: {summary['img_size']}x{summary['img_size']}")
    lines.append("")
    lines.append("## 3. Preprocessing & Augmentation")
    lines.append("- Grayscale conversion and normalization to 0-1")
    lines.append("- Augmentations: rotation, translation, zoom, contrast, Gaussian noise")
    lines.append("")
    lines.append("## 4. Models")
    lines.append("- DNN: 2 hidden layers (512, 256) with dropout")
    lines.append("- CNN: 2 convolution layers (32, 64) with pooling and dropout")
    lines.append("")
    lines.append("## 5. Evaluation")
    for result in summary["results"]:
        lines.append(
            f"- {result['model'].upper()} test accuracy: {result['test_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("## 6. Comparative Analysis")
    lines.append(
        "CNN typically performs better on image data due to spatial feature learning, while DNN is simpler but less robust."
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_report_pdf(md_path: Path, pdf_path: Path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            pdf.set_font("Helvetica", size=16)
            pdf.cell(0, 10, line[2:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=12)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", size=14)
            pdf.cell(0, 8, line[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=12)
        elif line.startswith("- "):
            pdf.multi_cell(pdf.epw, 6, f"- {line[2:]}")
        else:
            pdf.multi_cell(pdf.epw, 6, line)

    pdf.output(str(pdf_path))


def main():
    outputs_dir = Path("outputs")
    summary_path = outputs_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("summary.json not found. Run run_all.py first.")

    summary = load_summary(summary_path)
    md_path = Path("report.md")
    pdf_path = Path("report.pdf")
    build_report_md(summary, md_path)
    build_report_pdf(md_path, pdf_path)

    print("Report generated: report.md and report.pdf")


if __name__ == "__main__":
    main()
