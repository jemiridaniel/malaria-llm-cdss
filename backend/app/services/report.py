"""
PDF clinical report generation using ReportLab.
"""
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Color constants ────────────────────────────────────────────────────────────
PRIMARY      = colors.HexColor("#0284C7")
LIGHT_BG     = colors.HexColor("#F0F9FF")
BORDER_COLOR = colors.HexColor("#E2E8F0")
TEXT         = colors.HexColor("#0F172A")
TEXT_MUTED   = colors.HexColor("#64748B")

STAGE_COLORS = {
    "Critical":   colors.HexColor("#DC2626"),
    "Stage_II":   colors.HexColor("#EA580C"),
    "Stage_I":    colors.HexColor("#D97706"),
    "No_Malaria": colors.HexColor("#16A34A"),
}
STAGE_LABELS = {
    "Critical":   "CRITICAL — Severe Malaria",
    "Stage_II":   "Stage II Malaria",
    "Stage_I":    "Stage I Malaria",
    "No_Malaria": "No Malaria Detected",
}


def _section_label(text: str) -> Paragraph:
    return Paragraph(
        text,
        ParagraphStyle(
            "SL",
            fontSize=8,
            fontName="Helvetica-Bold",
            textColor=TEXT_MUTED,
            spaceBefore=12,
            spaceAfter=5,
        ),
    )


def _body(text: str, bold: bool = False, text_color: Any = None) -> Paragraph:
    return Paragraph(
        text,
        ParagraphStyle(
            "Body",
            fontSize=10,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            textColor=text_color or TEXT,
            leading=15,
        ),
    )


def _info_table(rows: List[List[str]]) -> Table:
    col_widths = [3.5 * cm, 7.5 * cm, 3 * cm, 3.5 * cm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(
        TableStyle([
            ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",     (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",    (0, 0), (0, -1),  TEXT_MUTED),
            ("TEXTCOLOR",    (2, 0), (2, -1),  TEXT_MUTED),
            ("TEXTCOLOR",    (1, 0), (-1, -1), TEXT),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_BG, colors.white]),
            ("BOX",          (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID",    (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ])
    )
    return t


def _boxed(content: Paragraph, bg: Any, border: Any) -> Table:
    t = Table([[content]], colWidths=None)  # width set at build time via doc.width
    t.setStyle(
        TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), bg),
            ("BOX",          (0, 0), (-1, -1), 0.75, border),
            ("TOPPADDING",   (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
            ("LEFTPADDING",  (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ])
    )
    return t


def generate_report(data: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2.5 * cm,
    )

    story = []

    # ── Header banner ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "HTitle", fontSize=15, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER, leading=20,
    )
    sub_style = ParagraphStyle(
        "HSub", fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#BAE6FD"), alignment=TA_CENTER, leading=14,
    )
    banner = Table(
        [
            [Paragraph("Malaria Clinical Decision Support Report", title_style)],
            [Paragraph("AI-Assisted Diagnosis  |  For Clinical Decision Support Only", sub_style)],
        ],
        colWidths=[doc.width],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), PRIMARY),
        ("TOPPADDING",   (0, 0), (-1, 0),  14),
        ("BOTTOMPADDING",(0, -1), (-1, -1), 14),
        ("LEFTPADDING",  (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.5 * cm))

    # ── Patient Information ────────────────────────────────────────────────────
    story.append(_section_label("PATIENT INFORMATION"))
    patient_name = data.get("patient_name") or "Not provided"
    patient_id   = data.get("patient_id")   or "Not provided"
    timestamp    = data.get("timestamp")    or datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(_info_table([
        ["Patient Name:", patient_name,  "Patient ID:", patient_id],
        ["Report Date/Time:", timestamp, "",            ""],
    ]))

    # ── Demographics ───────────────────────────────────────────────────────────
    story.append(_section_label("DEMOGRAPHICS"))
    demo = data.get("demographics", {})
    story.append(_info_table([
        ["Age:",        str(demo.get("age", "Unknown")),                       "Sex:",       demo.get("sex", "Unknown")],
        ["Pregnant:",   demo.get("pregnant", "no").capitalize(),               "Genotype:",  demo.get("genotype", "AA")],
        ["Blood Type:", demo.get("blood_type", "Unknown"),                     "RDT Result:",demo.get("rdt_result", "not_done").replace("_", " ").title()],
        ["Microscopy:", demo.get("microscopy_result", "not_done").replace("_", " ").title(), "", ""],
    ]))

    # ── Diagnosis ──────────────────────────────────────────────────────────────
    story.append(_section_label("DIAGNOSIS RESULT"))
    severity    = data.get("severity_stage", "Unknown")
    stage_color = STAGE_COLORS.get(severity, PRIMARY)
    stage_label = STAGE_LABELS.get(severity, severity)
    confidence  = data.get("confidence", 0)

    badge_style = ParagraphStyle(
        "Badge", fontSize=12, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER, leading=16,
    )
    conf_style = ParagraphStyle(
        "Conf", fontSize=10, fontName="Helvetica",
        textColor=colors.white, alignment=TA_CENTER, leading=16,
    )
    diag_row = Table(
        [[Paragraph(stage_label, badge_style), Paragraph(f"Confidence: {confidence}%", conf_style)]],
        colWidths=[doc.width * 0.65, doc.width * 0.35],
    )
    diag_row.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (0, 0), stage_color),
        ("BACKGROUND",   (1, 0), (1, 0), colors.HexColor("#1E293B")),
        ("TOPPADDING",   (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 11),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(diag_row)
    story.append(Spacer(1, 0.25 * cm))
    story.append(_body(data.get("diagnosis_text", ""), bold=True))

    # ── Reported Symptoms ──────────────────────────────────────────────────────
    story.append(_section_label("REPORTED SYMPTOMS"))
    symptoms: List[str] = data.get("positive_symptoms", [])
    if symptoms:
        story.append(_body(f"{len(symptoms)} symptom(s) reported"))
        story.append(Spacer(1, 0.2 * cm))
        rows = []
        for i in range(0, len(symptoms), 3):
            chunk = symptoms[i : i + 3]
            while len(chunk) < 3:
                chunk.append("")
            rows.append([f"• {s.title()}" if s else "" for s in chunk])
        sym_t = Table(rows, colWidths=[doc.width / 3] * 3)
        sym_t.setStyle(TableStyle([
            ("FONTSIZE",       (0, 0), (-1, -1), 9.5),
            ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
            ("TEXTCOLOR",      (0, 0), (-1, -1), TEXT),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_BG, colors.white]),
            ("BOX",            (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID",      (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ]))
        story.append(sym_t)
    else:
        story.append(_body("No symptoms reported."))

    # ── AI Clinical Explanation ────────────────────────────────────────────────
    story.append(_section_label("AI CLINICAL EXPLANATION"))
    reason_t = Table(
        [[_body(data.get("clinical_reasoning") or "No explanation available.")]],
        colWidths=[doc.width],
    )
    reason_t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), LIGHT_BG),
        ("BOX",          (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(reason_t)
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        f"Model: {data.get('model_used', 'unknown')}",
        ParagraphStyle("Mi", fontSize=8, fontName="Helvetica-Oblique", textColor=TEXT_MUTED),
    ))

    # ── Treatment Recommendation ───────────────────────────────────────────────
    story.append(_section_label("TREATMENT RECOMMENDATION"))
    is_critical  = severity == "Critical"
    presc_bg     = colors.HexColor("#FEF2F2") if is_critical else LIGHT_BG
    presc_border = colors.HexColor("#FECACA") if is_critical else BORDER_COLOR
    presc_color  = colors.HexColor("#991B1B") if is_critical else TEXT
    presc_t = Table(
        [[_body(data.get("prescription", ""), bold=is_critical, text_color=presc_color)]],
        colWidths=[doc.width],
    )
    presc_t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), presc_bg),
        ("BOX",          (0, 0), (-1, -1), 1.0, presc_border),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(presc_t)

    # ── Footer disclaimer ──────────────────────────────────────────────────────
    story.append(Spacer(1, 0.7 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-powered clinical decision support tool. "
        "Final diagnosis must be confirmed by a licensed clinician. "
        "This document does not constitute medical advice and should not replace professional medical consultation.",
        ParagraphStyle(
            "Disc", fontSize=7.5, fontName="Helvetica-Oblique",
            textColor=TEXT_MUTED, alignment=TA_CENTER, leading=11,
        ),
    ))

    doc.build(story)
    return buffer.getvalue()
