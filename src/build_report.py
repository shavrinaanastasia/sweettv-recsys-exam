"""Build the report PDF from report/report_text.md with reportlab.

Markup: '# ' title, '## ' section, '### ' subsection, blank-line paragraphs, '- ' bullets,
pipe tables, '![caption](file.png)' figures, '<<META key: value>>' title-block fields,
**bold**, *italic*, `code`. '<<TODO: ...>>' markers are rendered in red and the output is
then named *_DRAFT.pdf so an unfinished report cannot be mistaken for the final one.
'<<PAGEBREAK>>' forces a new page (used before References).
"""
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from common import ROOT

REPORT = ROOT / "report"
FONTS = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("TNR", FONTS + "times.ttf"))
pdfmetrics.registerFont(TTFont("TNR-B", FONTS + "timesbd.ttf"))
pdfmetrics.registerFont(TTFont("TNR-I", FONTS + "timesi.ttf"))
pdfmetrics.registerFont(TTFont("TNR-BI", FONTS + "timesbi.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FONTS + "consola.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily

registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")

BASE = 9.6
S = {
    "title": ParagraphStyle("title", fontName="TNR-B", fontSize=15, leading=18, alignment=TA_CENTER, spaceAfter=4),
    "meta": ParagraphStyle("meta", fontName="TNR", fontSize=9, leading=11, alignment=TA_CENTER),
    "h2": ParagraphStyle("h2", fontName="TNR-B", fontSize=11, leading=13, spaceBefore=7, spaceAfter=2),
    "h3": ParagraphStyle("h3", fontName="TNR-BI", fontSize=BASE, leading=BASE + 2, spaceBefore=4, spaceAfter=1),
    "p": ParagraphStyle("p", fontName="TNR", fontSize=BASE, leading=BASE + 2.2, alignment=TA_JUSTIFY, spaceAfter=3),
    "li": ParagraphStyle("li", fontName="TNR", fontSize=BASE, leading=BASE + 2.2, leftIndent=11, bulletIndent=2,
                         alignment=TA_JUSTIFY, spaceAfter=1),
    "cap": ParagraphStyle("cap", fontName="TNR", fontSize=8, leading=9.5, alignment=TA_CENTER, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="TNR", fontSize=7.6, leading=8.8),
    "cellb": ParagraphStyle("cellb", fontName="TNR-B", fontSize=7.6, leading=8.8),
    "ref": ParagraphStyle("ref", fontName="TNR", fontSize=8.6, leading=10.4, leftIndent=14, firstLineIndent=-14,
                          spaceAfter=2),
}
todo_count = 0


def inline(t):
    global todo_count
    t = t.replace("&", "&amp;").replace("<<", "\x00").replace(">>", "\x01").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font name="Mono" size="8">\1</font>', t)

    def todo(m):
        global todo_count
        todo_count += 1
        return f'<font color="#c0392b"><b>[TODO: {m.group(1)}]</b></font>'

    t = re.sub(r"\x00TODO:\s*(.+?)\x01", todo, t)
    return t.replace("\x00", "&lt;&lt;").replace("\x01", "&gt;&gt;")


def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    data = [[Paragraph(inline(c), S["cellb"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    # column widths proportional to the longest plain text in each column (wrapping only the long ones)
    plain = lambda c: re.sub(r"[*`]", "", c)
    longest = [min(max(len(plain(r[j])) for r in rows), 60) + 4 for j in range(len(rows[0]))]
    usable = A4[0] - 3.8 * cm
    t = Table(data, repeatRows=1, hAlign="CENTER", colWidths=[usable * w / sum(longest) for w in longest])
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build(src, out_stem):
    story, meta = [], []
    lines = []
    for l in src.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"<<INCLUDE (.+)>>", l.strip())
        lines += (REPORT / m.group(1)).read_text(encoding="utf-8").splitlines() if m else [l]
    i, para = 0, []

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(para)), S["p"]))
            para.clear()

    while i < len(lines):
        l = lines[i]
        if not l.strip():
            flush()
        elif l.startswith("<<META"):
            meta.append(l.rstrip()[len("<<META"):-2].strip())  # drop only the closing '>>'
        elif l.strip() == "<<PAGEBREAK>>":
            flush()
            story.append(PageBreak())
        elif l.startswith("# "):
            flush()
            story.append(Paragraph(inline(l[2:]), S["title"]))
            for m in meta:
                story.append(Paragraph(inline(m), S["meta"]))
            story.append(Spacer(1, 6))
        elif l.startswith("## "):
            flush()
            story.append(Paragraph(inline(l[3:]), S["h2"]))
        elif l.startswith("### "):
            flush()
            story.append(Paragraph(inline(l[4:]), S["h3"]))
        elif l.startswith("- "):
            flush()
            story.append(Paragraph(inline(l[2:]), S["li"], bulletText="•"))
        elif re.match(r"\[\d+\] ", l):
            flush()
            story.append(Paragraph(inline(l), S["ref"]))
        elif l.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            cap = None
            if i < len(lines) and lines[i].startswith("Table"):
                cap = lines[i]
                i += 1
            parts = ([Paragraph(inline(cap), S["cap"])] if cap else []) + [table(block), Spacer(1, 5)]
            story.append(KeepTogether(parts))
            continue
        elif l.startswith("!["):
            flush()
            m = re.match(r"!\[(.*)\]\((.+?)\)(?:\{w=([\d.]+)\})?", l)
            img = Image(str(REPORT / m.group(2)))
            w = float(m.group(3) or 8.6) * cm
            img.drawHeight, img.drawWidth = img.imageHeight * w / img.imageWidth, w
            story.append(KeepTogether([img, Paragraph(inline(m.group(1)), S["cap"])]))
        else:
            para.append(l.strip())
        i += 1
    flush()

    name = out_stem + ("_DRAFT" if todo_count else "") + ".pdf"
    doc = SimpleDocTemplate(str(REPORT / name), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm, title=out_stem,
                            author="Anastasia Shavrina")

    def footer(canvas, d):
        canvas.setFont("TNR", 8)
        canvas.drawCentredString(A4[0] / 2, 0.9 * cm, str(d.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return name


if __name__ == "__main__":
    out = build(REPORT / "report_text.md", sys.argv[1] if len(sys.argv) > 1 else "shavrina_anastasia_exam_report")
    print(out, "| TODO markers:", todo_count)
