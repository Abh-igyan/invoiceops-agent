"""Synthetic PDF invoice renderer with multi-layout templates and scan noise."""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from invoiceops.schemas import InvoiceDocument

LayoutStyle = Literal["classic_corporate", "modern_minimal", "legacy_tabular"]


def render_invoice_pdf(
    invoice: InvoiceDocument,
    output_path: Path,
    layout_style: LayoutStyle = "classic_corporate",
    is_noisy_scan: bool = False,
    skew_degrees: float = 0.0,
    blur_radius: float = 0.0,
    rng_seed: int = 0,
) -> None:
    """Render an `InvoiceDocument` to PDF, applying optional scan skew/blur perturbations."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_noisy_scan:
        _render_noisy_scanned_pdf(
            invoice=invoice,
            output_path=output_path,
            layout_style=layout_style,
            skew_degrees=skew_degrees,
            blur_radius=blur_radius,
            rng_seed=rng_seed,
        )
    else:
        _render_digital_pdf(invoice=invoice, output_path=output_path, layout_style=layout_style)


def _render_digital_pdf(
    invoice: InvoiceDocument,
    output_path: Path,
    layout_style: LayoutStyle,
) -> None:
    """Render crisp vector PDF using ReportLab with one of three distinct layouts."""
    c = canvas.Canvas(str(output_path), pagesize=LETTER)
    width, height = LETTER

    if layout_style == "classic_corporate":
        c.setFillColor(colors.HexColor("#1B365D"))
        c.rect(40, height - 95, width - 80, 55, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(55, height - 68, f"COMMERCIAL INVOICE  |  {invoice.vendor_name}")
        c.setFont("Helvetica", 10)
        c.drawString(55, height - 84, f"Vendor ID: {invoice.vendor_id}")
    elif layout_style == "modern_minimal":
        c.setFillColor(colors.HexColor("#111827"))
        c.setFont("Helvetica-Bold", 22)
        c.drawString(45, height - 65, "INVOICE")
        c.setFont("Helvetica", 11)
        c.drawRightString(width - 45, height - 60, invoice.vendor_name)
        c.drawRightString(width - 45, height - 76, f"Supplier Code: {invoice.vendor_id}")
        c.setStrokeColor(colors.HexColor("#D1D5DB"))
        c.line(45, height - 90, width - 45, height - 90)
    else:  # legacy_tabular
        c.setStrokeColor(colors.black)
        c.rect(40, height - 95, width - 80, 55, fill=0, stroke=1)
        c.setFillColor(colors.black)
        c.setFont("Courier-Bold", 15)
        c.drawString(50, height - 63, f"AP BILLING DOC // {invoice.vendor_name.upper()}")
        c.setFont("Courier", 10)
        c.drawString(50, height - 82, f"VENDOR_MASTER_ID={invoice.vendor_id}")

    body_font = "Courier" if layout_style == "legacy_tabular" else "Helvetica"
    bold_font = "Courier-Bold" if layout_style == "legacy_tabular" else "Helvetica-Bold"

    c.setFillColor(colors.black)
    c.setFont(bold_font, 10)
    y = height - 125
    po_display = invoice.po_id if invoice.po_id else "NONE (NO PO REFERENCED)"
    meta_left = [
        f"Invoice Number: {invoice.invoice_number}",
        f"Internal Case Ref: {invoice.invoice_id}",
        f"Purchase Order: {po_display}",
    ]
    meta_right = [
        f"Invoice Date: {invoice.invoice_date}",
        f"Due Date: {invoice.due_date}",
        f"Billing Currency: {invoice.currency}",
    ]
    for left_txt, right_txt in zip(meta_left, meta_right, strict=True):
        c.setFont(body_font, 10)
        c.drawString(45, y, left_txt)
        c.drawString(330, y, right_txt)
        y -= 16

    y -= 16
    c.setFont(bold_font, 10)
    c.drawString(45, y, "SKU")
    c.drawString(130, y, "Description")
    c.drawRightString(370, y, "Qty")
    c.drawRightString(460, y, f"Unit ({invoice.currency})")
    c.drawRightString(560, y, f"Line Total ({invoice.currency})")
    y -= 6
    c.line(45, y, 565, y)
    y -= 16

    c.setFont(body_font, 10)
    for line in invoice.lines:
        c.drawString(45, y, line.sku)
        c.drawString(130, y, line.description[:34])
        c.drawRightString(370, y, str(line.quantity))
        c.drawRightString(460, y, f"{line.unit_price:,.2f}")
        c.drawRightString(560, y, f"{line.line_total:,.2f}")
        y -= 16

    y -= 10
    c.line(340, y, 565, y)
    y -= 18
    c.drawString(350, y, "Subtotal:")
    c.drawRightString(560, y, f"{invoice.currency} {invoice.subtotal:,.2f}")
    y -= 16
    c.drawString(350, y, f"Tax ({invoice.tax_rate * 100:.2f}%):")
    c.drawRightString(560, y, f"{invoice.currency} {invoice.tax_amount:,.2f}")
    y -= 18
    c.setFont(bold_font, 11)
    c.drawString(350, y, "TOTAL DUE:")
    c.drawRightString(560, y, f"{invoice.currency} {invoice.total_amount:,.2f}")

    if invoice.notes:
        y -= 40
        c.setFont(bold_font, 9)
        c.drawString(45, y, "Supplier Notes / Remittance Remarks:")
        y -= 14
        c.setFont(body_font, 9)
        for chunk in _wrap_text(invoice.notes, width_chars=82):
            c.drawString(45, y, chunk)
            y -= 13

    c.save()


def _render_noisy_scanned_pdf(
    invoice: InvoiceDocument,
    output_path: Path,
    layout_style: LayoutStyle,
    skew_degrees: float,
    blur_radius: float,
    rng_seed: int,
) -> None:
    """Render a rasterized scan with controlled skew, Gaussian blur, and paper speckle."""
    img_w, img_h = 1224, 1584  # 144 DPI US Letter
    img = Image.new("RGB", (img_w, img_h), color=(250, 249, 245))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=20)
    font_bold = ImageFont.load_default(size=24)
    font_small = ImageFont.load_default(size=17)

    header_bg = (35, 55, 85) if layout_style == "classic_corporate" else (235, 235, 235)
    header_fg = (255, 255, 255) if layout_style == "classic_corporate" else (15, 15, 15)
    draw.rectangle((70, 70, img_w - 70, 185), fill=header_bg, outline=(30, 30, 30), width=2)
    draw.text(
        (95, 92),
        f"SCANNED SUPPLIER INVOICE - {invoice.vendor_name}",
        fill=header_fg,
        font=font_bold,
    )
    draw.text(
        (95, 136),
        f"Vendor ID: {invoice.vendor_id}  |  Layout: {layout_style}",
        fill=header_fg,
        font=font,
    )

    po_display = invoice.po_id if invoice.po_id else "NONE"
    y = 225
    meta_rows = [
        (f"Invoice Number: {invoice.invoice_number}", f"Invoice Date: {invoice.invoice_date}"),
        (f"Case Reference: {invoice.invoice_id}", f"Due Date: {invoice.due_date}"),
        (f"Purchase Order: {po_display}", f"Currency: {invoice.currency}"),
    ]
    for left_col, right_col in meta_rows:
        draw.text((80, y), left_col, fill=(20, 20, 20), font=font)
        draw.text((660, y), right_col, fill=(20, 20, 20), font=font)
        y += 34

    y += 24
    draw.line((80, y, img_w - 80, y), fill=(40, 40, 40), width=2)
    y += 14
    draw.text((80, y), "SKU", fill=(0, 0, 0), font=font)
    draw.text((240, y), "Description", fill=(0, 0, 0), font=font)
    draw.text((690, y), "Qty", fill=(0, 0, 0), font=font)
    draw.text((800, y), f"Unit ({invoice.currency})", fill=(0, 0, 0), font=font)
    draw.text((980, y), f"Line ({invoice.currency})", fill=(0, 0, 0), font=font)
    y += 32
    draw.line((80, y, img_w - 80, y), fill=(40, 40, 40), width=2)
    y += 16

    for line in invoice.lines:
        draw.text((80, y), line.sku, fill=(20, 20, 20), font=font)
        draw.text((240, y), line.description[:28], fill=(20, 20, 20), font=font)
        draw.text((690, y), str(line.quantity), fill=(20, 20, 20), font=font)
        draw.text((800, y), f"{line.unit_price:,.2f}", fill=(20, 20, 20), font=font)
        draw.text((980, y), f"{line.line_total:,.2f}", fill=(20, 20, 20), font=font)
        y += 34

    y += 20
    draw.line((680, y, img_w - 80, y), fill=(40, 40, 40), width=2)
    y += 18
    draw.text(
        (690, y),
        f"Subtotal: {invoice.currency} {invoice.subtotal:,.2f}",
        fill=(15, 15, 15),
        font=font,
    )
    y += 32
    draw.text(
        (690, y),
        f"Tax ({invoice.tax_rate * 100:.2f}%): {invoice.currency} {invoice.tax_amount:,.2f}",
        fill=(15, 15, 15),
        font=font,
    )
    y += 36
    draw.text(
        (690, y),
        f"TOTAL DUE: {invoice.currency} {invoice.total_amount:,.2f}",
        fill=(0, 0, 0),
        font=font_bold,
    )

    if invoice.notes:
        y += 65
        draw.text((80, y), "Supplier Notes / Remarks:", fill=(15, 15, 15), font=font)
        y += 30
        for chunk in _wrap_text(invoice.notes, width_chars=74):
            draw.text((80, y), chunk, fill=(30, 30, 30), font=font_small)
            y += 26

    # Deterministic scan speckle
    rng = random.Random(rng_seed)
    for _ in range(900):
        px = rng.randrange(0, img_w)
        py = rng.randrange(0, img_h)
        shade = rng.randrange(150, 225)
        img.putpixel((px, py), (shade, shade, shade))

    if abs(skew_degrees) > 0.01:
        img = img.rotate(
            skew_degrees,
            resample=Image.Resampling.BILINEAR,
            expand=False,
            fillcolor=(244, 242, 238),
        )
    if blur_radius > 0.01:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    buf.seek(0)

    c = canvas.Canvas(str(output_path), pagesize=LETTER)
    width, height = LETTER
    c.drawImage(ImageReader(buf), 0, 0, width=width, height=height)
    c.save()


def _wrap_text(text: str, width_chars: int = 80) -> list[str]:
    """Wrap plain text into bounded character lines without breaking words."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    cur_len = 0
    for word in words:
        add_len = len(word) + (1 if current else 0)
        if cur_len + add_len > width_chars and current:
            lines.append(" ".join(current))
            current = [word]
            cur_len = len(word)
        else:
            current.append(word)
            cur_len += add_len
    if current:
        lines.append(" ".join(current))
    return lines
