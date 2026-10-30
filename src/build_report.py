"""Build the report PDF from report/report_text.md with reportlab, styled after the course's
sample_homework.pdf (Calibri, blue section headings, italic centred title block, bold labels
on their own line, blue banded tables, italic "Figure N:" captions).

Markup: '# ' title, '## ' section, '### ' subsection, blank-line paragraphs, '- ' bullets,
pipe tables (+ 'Table N: ...' caption line), '![Figure N: caption](file.png){w=cm}',
'<<META text>>' title-block lines, '<<INCLUDE file>>', '<<PAGEBREAK>>', '%%' comment lines,
**bold**, *italic*, `code`. A paragraph that starts with '**Label.**' is split into a bold
label line and the text. '<<TODO: ...>>' markers render in red and the output is then named
*_DRAFT.pdf so an unfinished report cannot be mistaken for the final one.
"""
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from common import ROOT

REPORT = ROOT / "report"
FONTS = "C:/Windows/Fonts/"
for name, f in [("Cal", "calibri.ttf"), ("Cal-B", "calibrib.ttf"), ("Cal-I", "calibrii.ttf"),
                ("Cal-BI", "calibriz.ttf"), ("Mono", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(name, FONTS + f))
registerFontFamily("Cal", normal="Cal", bold="Cal-B", italic="Cal-I", boldItalic="Cal-BI")

BLUE = colors.HexColor("#2F5496")      # Word "Heading 1" blue, as in the sample
GRID = colors.HexColor("#8EAADB")
BAND = colors.HexColor("#D9E2F3")
BASE = 9.2
S = {  # headings and labels keep with the following block, so none is left alone at a page bottom
    "title": ParagraphStyle("title", fontName="Cal-B", fontSize=17, leading=21, alignment=TA_CENTER, spaceAfter=6),
    "meta": ParagraphStyle("meta", fontName="Cal-I", fontSize=9.5, leading=12, alignment=TA_CENTER),
    "h2": ParagraphStyle("h2", fontName="Cal-B", fontSize=14, leading=17, textColor=BLUE, spaceBefore=7, spaceAfter=2),
    "h3": ParagraphStyle("h3", fontName="Cal-B", fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=4, spaceAfter=1),
    "label": ParagraphStyle("label", fontName="Cal-B", fontSize=BASE, leading=BASE + 2.6, spaceBefore=3, spaceAfter=1),
    "p": ParagraphStyle("p", fontName="Cal", fontSize=BASE, leading=BASE + 2.6, alignment=TA_LEFT, spaceAfter=3),
    "li": ParagraphStyle("li", fontName="Cal", fontSize=BASE, leading=BASE + 2.6, leftIndent=14, bulletIndent=4,
                         spaceAfter=1.5),
    "cap": ParagraphStyle("cap", fontName="Cal-I", fontSize=9, leading=11, alignment=TA_LEFT, spaceBefore=2,
                          spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Cal", fontSize=8, leading=9.2),
    "cellb": ParagraphStyle("cellb", fontName="Cal-B", fontSize=8, leading=9.2),
    "ref": ParagraphStyle("ref", fontName="Cal", fontSize=9.5, leading=12, spaceAfter=5),
}
for k in ("h2", "h3", "label"):
    S[k].keepWithNext = 1
todo_count = 0


def inline(t):
    global todo_count
    t = t.replace("&", "&amp;").replace("<<", "\x00").replace(">>", "\x01").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font name="Mono" size="8.5">\1</font>', t)

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
    plain = lambda c: re.sub(r"[*`]", "", c)
    longest = [min(max(len(plain(r[j])) for r in rows), 60) + 4 for j in range(len(rows[0]))]
    usable = A4[0] - 4.0 * cm
    widths = [usable * w / sum(longest) for w in longest]
    # never narrower than the longest single word, so ids and categories do not break mid-word
    word = [max(pdfmetrics.stringWidth(w, "Cal-B", 8) for r in rows for w in plain(r[j]).split() or [""]) + 8
            for j in range(len(rows[0]))]
    short = [max(0.0, word[j] - widths[j]) for j in range(len(widths))]
    if sum(short):
        slack = [max(0.0, widths[j] - word[j]) for j in range(len(widths))]
        take = sum(short) / max(sum(slack), 1e-9)
        widths = [max(word[j], widths[j] - slack[j] * take) for j in range(len(widths))]
    t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=widths)
    style = [("GRID", (0, 0), (-1, -1), 0.5, GRID), ("LINEBELOW", (0, 0), (-1, 0), 1.5, colors.HexColor("#4472C4")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 1), ("LEFTPADDING", (0, 0), (-1, -1), 3),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    style += [("BACKGROUND", (0, r), (-1, r), BAND) for r in range(1, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def read_lines(path):
    out = []
    for l in path.read_text(encoding="utf-8-sig").splitlines():
        m = re.fullmatch(r"<<INCLUDE (.+)>>", l.strip())
        if m:
            out += read_lines(REPORT / m.group(1))
        elif not l.startswith("%%"):
            out.append(l)
    return out


def build(src, out_stem):
    story, meta = [], []
    lines = read_lines(src)
    i, para = 0, []

    def flush():
        if para:
            text = " ".join(para)
            m = re.match(r"\*\*([^*]+?[.:?])\*\*\s+(.*)", text)
            if m:  # sample style: bold label on its own line, text below
                story.append(Paragraph(inline(m.group(1)), S["label"]))
                text = m.group(2)
            story.append(Paragraph(inline(text), S["p"]))
            para.clear()

    while i < len(lines):
        l = lines[i]
        if not l.strip():
            flush()
        elif l.startswith("<<META"):
            meta.append(l.rstrip()[len("<<META"):-2].strip())
        elif l.strip() == "<<PAGEBREAK>>":
            flush()
            story.append(PageBreak())
        elif l.startswith("# "):
            flush()
            story.append(Paragraph(inline(l[2:]), S["title"]))
            for m in meta:
                story.append(Paragraph(inline(m), S["meta"]))
            story.append(Spacer(1, 8))
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
            parts = [table(block)] + ([Paragraph(inline(cap), S["cap"])] if cap else [Spacer(1, 6)])
            if len(block) <= 12:
                story.append(KeepTogether(parts))
            else:  # long table: may split across pages (header row repeats) instead of leaving a gap
                story += parts
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
    doc = SimpleDocTemplate(str(REPORT / name), pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title=out_stem, author="Anastasia Shavrina")

    def footer(canvas, d):
        canvas.setFont("Cal", 8.5)
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm, str(d.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return name


if __name__ == "__main__":
    out = build(REPORT / "report_text.md", sys.argv[1] if len(sys.argv) > 1 else "shavrina_anastasia_exam_report")
    print(out, "| TODO markers:", todo_count)
