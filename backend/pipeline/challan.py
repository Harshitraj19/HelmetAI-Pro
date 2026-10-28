"""
Auto-generated e-Challan PDF  (Novelty #5, enforcement output)

One A4 page: header, offence table with fine amounts, evidence images, plate,
integrity block (record / chain hash) and a QR code that encodes the verification
URL + record hash, so anyone can check the challan against the evidence log.

Fine amounts follow the Motor Vehicles (Amendment) Act 2019 headline values and are
configurable - they are indicative, for demonstration only.
"""
from __future__ import annotations

import io
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

OFFENCES = {
    "NO_HELMET":       ("Riding without protective headgear", "Sec 129 r/w 194D MVA", 1000),
    "INVALID_HELMET":  ("Non-standard headgear (construction hard-hat)", "Sec 129 r/w 194D MVA", 1000),
    "HELMET_NOT_WORN": ("Helmet carried but not worn", "Sec 129 r/w 194D MVA", 1000),
    "TRIPLE_RIDING":   ("More than one pillion rider", "Sec 128 r/w 194C MVA", 1000),
}


def build_challan(rec: dict, evidence_root: Path, verify_url: str = "") -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    m = 15 * mm

    # header band
    c.setFillColor(colors.HexColor("#0f172a"))
    c.rect(0, H - 32 * mm, W, 32 * mm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(m, H - 15 * mm, "e-CHALLAN  |  Traffic Violation Notice")
    c.setFont("Helvetica", 9)
    c.drawString(m, H - 22 * mm, "Generated automatically by HelmetAI-Pro  -  AI evidence, pending officer review")
    c.drawRightString(W - m, H - 15 * mm, f"Challan No: HAI-{rec['id']:06d}")
    c.drawRightString(W - m, H - 22 * mm, f"Date/Time (UTC): {str(rec['created_at'])[:19]}")

    # details
    y = H - 44 * mm
    c.setFillColor(colors.black)
    rows = [
        ("Vehicle number", rec.get("plate") or "NOT READ - manual verification needed"),
        ("Plate read confidence", f"{rec.get('plate_confidence') or 0:.0%}"),
        ("Location / camera", f"{rec.get('camera_id') or 'CAM-01'}   (source: {rec.get('source') or '-'})"),
        ("Riders on vehicle", str(rec.get("riders") or "-")),
        ("Severity score", f"{rec.get('severity', 0)} / 100"),
    ]
    c.setFont("Helvetica", 10)
    for k, v in rows:
        c.setFillColor(colors.HexColor("#475569")); c.drawString(m, y, k)
        c.setFillColor(colors.black); c.drawString(m + 48 * mm, y, str(v))
        y -= 6.5 * mm

    # offences table
    y -= 3 * mm
    c.setFillColor(colors.HexColor("#e2e8f0")); c.rect(m, y - 2 * mm, W - 2 * m, 8 * mm, fill=1, stroke=0)
    c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 10)
    c.drawString(m + 2 * mm, y, "Offence"); c.drawString(m + 105 * mm, y, "Section"); c.drawRightString(W - m - 2 * mm, y, "Fine (INR)")
    c.setFont("Helvetica", 10)
    total = 0
    for v in (rec.get("violation_types") or "").split(","):
        if v not in OFFENCES:
            continue
        desc, sec, fine = OFFENCES[v]
        y -= 7 * mm
        c.drawString(m + 2 * mm, y, desc); c.drawString(m + 105 * mm, y, sec); c.drawRightString(W - m - 2 * mm, y, f"{fine:,}")
        total += fine
    y -= 8 * mm
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(W - m - 2 * mm, y, f"Total payable: INR {total:,}")

    # evidence images
    y -= 8 * mm
    c.setFont("Helvetica-Bold", 10); c.drawString(m, y, "Photographic evidence"); y -= 3 * mm
    img_h = 62 * mm
    x = m
    for key, maxw in (("frame_path", 100 * mm), ("bike_path", 45 * mm), ("plate_path", 30 * mm)):
        p = rec.get(key)
        if not p or not (evidence_root / p).exists():
            continue
        ir = ImageReader(str(evidence_root / p))
        iw, ih = ir.getSize()
        room = W - m - x                            # never run past the right margin
        if room < 15 * mm:
            break
        s = min(min(maxw, room) / iw, img_h / ih)
        c.drawImage(ir, x, y - ih * s, iw * s, ih * s)
        c.setStrokeColor(colors.HexColor("#94a3b8")); c.rect(x, y - ih * s, iw * s, ih * s)
        x += iw * s + 4 * mm
    y -= img_h + 8 * mm

    # integrity block + QR
    c.setFont("Helvetica-Bold", 10); c.drawString(m, y, "Evidence integrity (SHA-256 hash chain)")
    c.setFont("Courier", 7.5)
    c.drawString(m, y - 5 * mm, f"record : {rec['record_hash']}")
    c.drawString(m, y - 9 * mm, f"prev   : {rec['prev_hash']}")
    c.drawString(m, y - 13 * mm, f"chain  : {rec['chain_hash']}")
    c.setFont("Helvetica", 8)
    c.drawString(m, y - 19 * mm, "Any change to the images or details invalidates this hash. Scan the QR to verify.")
    qr = QrCodeWidget(f"{verify_url}?id={rec['id']}&h={rec['record_hash'][:16]}" if verify_url
                      else f"HAI-{rec['id']}|{rec['record_hash']}")
    b = qr.getBounds(); size = 32 * mm
    d = Drawing(size, size, transform=[size / (b[2] - b[0]), 0, 0, size / (b[3] - b[1]), 0, 0])
    d.add(qr)
    renderPDF.draw(d, c, W - m - size, y - 24 * mm)

    c.setFont("Helvetica-Oblique", 7.5); c.setFillColor(colors.HexColor("#64748b"))
    c.drawString(m, 12 * mm, "Indicative demo document. Fines per Motor Vehicles (Amendment) Act 2019 headline values; "
                             "issue only after human verification.")
    c.showPage(); c.save()
    return buf.getvalue()
