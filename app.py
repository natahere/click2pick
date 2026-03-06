"""
DocExtract — AI-powered document data extraction
Hosted on Hugging Face Spaces (Gradio)
"""

import gradio as gr
import anthropic
import base64
import json
import io
import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw

# ── Constants ─────────────────────────────────────────────────────────────────

PRESETS = {
    "📄 Invoice": [
        "Vendor Name", "Invoice Number", "Invoice Date", "Due Date",
        "Subtotal", "Tax Amount", "Total Amount", "Payment Terms", "PO Number"
    ],
    "🪪 ID / Passport": [
        "Full Name", "Date of Birth", "Gender", "Document Number",
        "Nationality", "Issue Date", "Expiry Date", "Place of Birth", "MRZ Line"
    ],
    "🧾 Receipt": [
        "Store Name", "Store Address", "Date", "Time",
        "Items", "Subtotal", "Tax", "Total", "Payment Method", "Transaction ID"
    ],
    "📝 Contract": [
        "Party A", "Party B", "Contract Date", "Effective Date",
        "Contract Value", "Duration", "Jurisdiction", "Governing Law"
    ],
    "🔑 Generic Key-Value": [],   # auto-extracted by Claude
    "✏️ Custom": [],
}

DEFAULT_PRESET = "📄 Invoice"

# HF Spaces can store a secret called ANTHROPIC_API_KEY
_ENV_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Helpers ───────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode()


def pdf_to_pil_pages(file_path: str) -> list[Image.Image]:
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


def load_document(file_obj) -> tuple[list[Image.Image], str]:
    """Load uploaded file → list of PIL pages + status message."""
    if file_obj is None:
        return [], "No file uploaded."
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        pages = pdf_to_pil_pages(path)
        return pages, f"✅ PDF loaded — {len(pages)} page(s)"
    elif ext in ("png", "jpg", "jpeg", "webp", "bmp", "tiff"):
        img = Image.open(path).convert("RGB")
        return [img], "✅ Image loaded"
    else:
        return [], f"❌ Unsupported file type: .{ext}"


def get_client(api_key: str):
    key = api_key.strip() or _ENV_KEY
    if not key:
        raise ValueError("Please enter your Anthropic API key.")
    return anthropic.Anthropic(api_key=key)


# ── Core extraction functions ─────────────────────────────────────────────────

def extract_region(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
                   field_name: str, api_key: str) -> str:
    """OCR a cropped region with Claude Vision."""
    crop = img.crop((x1, y1, x2, y2))
    client = get_client(api_key)
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": pil_to_b64(crop)}},
                {"type": "text", "text": (
                    f"Extract ONLY the text visible in this image region. "
                    f"This is for the field '{field_name}'. "
                    "Return ONLY the raw text, no quotes, no explanation."
                )},
            ]
        }]
    )
    return resp.content[0].text.strip()


def auto_extract_fields(img: Image.Image, fields: list[str], api_key: str) -> dict:
    """Send full page to Claude, extract all fields at once."""
    client = get_client(api_key)
    is_kv = not fields  # Generic key-value mode

    if is_kv:
        prompt = (
            "You are a precise document parser. "
            "Extract ALL key-value pairs visible in this document. "
            "Return ONLY a flat JSON object: {\"key\": \"value\", ...}. "
            "No markdown, no explanation."
        )
    else:
        prompt = (
            f"You are a precise document parser. "
            f"Extract these fields from the document: {json.dumps(fields)}. "
            "Return ONLY a flat JSON object with those exact field names as keys. "
            "Use empty string for fields not found. No markdown, no explanation."
        )

    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": "image/png",
                                              "data": pil_to_b64(img)}},
                {"type": "text", "text": prompt},
            ]
        }]
    )
    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── State helpers ─────────────────────────────────────────────────────────────

def fields_from_preset(preset_name: str) -> str:
    fields = PRESETS.get(preset_name, [])
    return "\n".join(fields)


def build_fields_df(fields: list[str], extracted: dict) -> list[list]:
    """Return rows for the dataframe: [Field, Value]"""
    if not fields:
        return [[k, v] for k, v in extracted.items()]
    return [[f, extracted.get(f, "")] for f in fields]


def annotate_page(img: Image.Image, boxes: list[dict]) -> Image.Image:
    """Draw extraction boxes on the image for visual feedback."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        draw.rectangle([x1, y1, x2, y2], outline="#F59E0B", width=3,
                       fill=(245, 158, 11, 30))
    return annotated


# ── CSS / Theme ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap');

:root {
    --bg:         #0c0c0f;
    --surface:    #141417;
    --surface2:   #1c1c21;
    --border:     #26262d;
    --accent:     #F59E0B;
    --accent-dim: #78480a;
    --text:       #e2dfd8;
    --muted:      #6b6b78;
    --success:    #34d399;
    --error:      #f87171;
}

/* Global reset */
body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; }
* { font-family: 'Lato', sans-serif !important; }
code, pre, .monospace { font-family: 'JetBrains Mono', monospace !important; }

/* Header */
#app-header {
    padding: 28px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
#app-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 32px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -1px;
    margin: 0;
}
#app-header h1 span { color: var(--accent); }
#app-header p {
    color: var(--muted);
    font-size: 14px;
    margin: 6px 0 0;
    font-weight: 300;
}

/* Panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
}
.panel-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
}

/* Gradio overrides */
.gr-button {
    background: var(--accent) !important;
    color: #0c0c0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    font-family: 'Syne', sans-serif !important;
    padding: 10px 18px !important;
    transition: all 0.18s !important;
    cursor: pointer !important;
}
.gr-button:hover { background: #d97706 !important; transform: translateY(-1px) !important; }
.gr-button.secondary {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}
.gr-button.secondary:hover { border-color: var(--accent) !important; }

/* Inputs */
input, textarea, select,
.gr-input, .gr-textarea, .gr-dropdown {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 14px !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(245,158,11,0.15) !important;
}

/* Image viewer */
.doc-viewer img { border-radius: 10px; border: 1px solid var(--border); }

/* Dataframe */
.gr-dataframe table {
    background: var(--surface2) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.gr-dataframe th {
    background: var(--surface) !important;
    color: var(--accent) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border) !important;
}
.gr-dataframe td {
    color: var(--text) !important;
    border-bottom: 1px solid var(--border) !important;
    font-size: 13px !important;
}
.gr-dataframe tr:hover td { background: var(--surface) !important; }

/* Status */
.status-ok  { color: var(--success) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }
.status-err { color: var(--error)   !important; font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important; }

/* Tab styling */
.gr-tab-nav button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 16px !important;
    background: transparent !important;
}
.gr-tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Accordion */
.gr-accordion { border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* JSON output */
.json-box {
    background: #080809;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #7ec8a4;
    white-space: pre-wrap;
    max-height: 360px;
    overflow-y: auto;
    line-height: 1.6;
}

/* Hide footer */
footer { display: none !important; }
"""

# ── Build Gradio UI ───────────────────────────────────────────────────────────

def build_ui():
    # Internal state (Gradio State)
    pages_state     = gr.State([])       # list[PIL.Image]
    extracted_state = gr.State({})       # dict field→value
    boxes_state     = gr.State([])       # drawn boxes for annotation
    page_idx_state  = gr.State(0)

    with gr.Blocks(css=CUSTOM_CSS, title="DocExtract") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="app-header">
            <h1>Doc<span>Extract</span></h1>
            <p>Upload any document · Draw regions to extract · Export structured data</p>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ══════════════════════════════════════════
            # LEFT COLUMN — Config + Fields + Results
            # ══════════════════════════════════════════
            with gr.Column(scale=1, min_width=320):

                # API Key
                gr.HTML('<div class="panel-label">⚙ Configuration</div>')
                api_key_input = gr.Textbox(
                    label="Anthropic API Key",
                    placeholder="sk-ant-api03-...",
                    type="password",
                    value=_ENV_KEY,
                    info="Get yours at console.anthropic.com"
                )

                gr.HTML('<hr style="border-color:#26262d;margin:16px 0">')

                # Template
                gr.HTML('<div class="panel-label">📋 Field Template</div>')
                preset_dd = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value=DEFAULT_PRESET,
                    label="",
                    show_label=False,
                    container=False,
                )
                fields_box = gr.Textbox(
                    label="Fields (one per line)",
                    value="\n".join(PRESETS[DEFAULT_PRESET]),
                    lines=9,
                    placeholder="Enter field names, one per line…\n(Leave empty for Generic Key-Value auto-detection)",
                )

                gr.HTML('<hr style="border-color:#26262d;margin:16px 0">')

                # Extraction controls
                gr.HTML('<div class="panel-label">🎯 Click-to-Extract</div>')
                active_field_dd = gr.Dropdown(
                    choices=PRESETS[DEFAULT_PRESET],
                    label="Target Field",
                    value=PRESETS[DEFAULT_PRESET][0] if PRESETS[DEFAULT_PRESET] else None,
                    info="Select a field, then draw a box on the document →"
                )

                with gr.Row():
                    extract_btn  = gr.Button("⚡ Auto-Extract All", variant="primary")
                    clear_btn    = gr.Button("🗑 Clear", variant="secondary")

                status_box = gr.HTML('<div class="status-ok"></div>')

                gr.HTML('<hr style="border-color:#26262d;margin:16px 0">')

                # Results table
                gr.HTML('<div class="panel-label">📊 Extracted Data</div>')
                results_df = gr.Dataframe(
                    headers=["Field", "Value"],
                    datatype=["str", "str"],
                    value=[],
                    interactive=True,
                    wrap=True,
                    row_count=(1, "dynamic"),
                )

                gr.HTML('<hr style="border-color:#26262d;margin:16px 0">')

                # Export
                gr.HTML('<div class="panel-label">💾 Export</div>')
                with gr.Row():
                    export_json_btn = gr.Button("⬇ JSON", variant="secondary", size="sm")
                    export_csv_btn  = gr.Button("⬇ CSV",  variant="secondary", size="sm")
                json_output  = gr.File(label="", visible=False)
                csv_output   = gr.File(label="", visible=False)

            # ══════════════════════════════════════════
            # RIGHT COLUMN — Document Viewer + Selector
            # ══════════════════════════════════════════
            with gr.Column(scale=2, min_width=500):
                gr.HTML('<div class="panel-label">📄 Document</div>')

                upload_btn = gr.File(
                    label="Upload PDF, PNG, JPG, WEBP, TIFF",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"],
                    type="filepath",
                )

                with gr.Row():
                    prev_btn     = gr.Button("◀ Prev", variant="secondary", size="sm", scale=1)
                    page_label   = gr.HTML('<div style="text-align:center;color:#6b6b78;font-size:12px;font-family:monospace;padding:8px">—</div>', )
                    next_btn     = gr.Button("Next ▶", variant="secondary", size="sm", scale=1)

                gr.HTML("""
                <div style="background:#13130f;border:1px solid #2a2600;border-left:3px solid #F59E0B;
                     border-radius:8px;padding:10px 14px;font-size:12px;color:#a89060;margin-bottom:12px">
                    <strong>How to use:</strong> Select a <em>Target Field</em> on the left →
                    draw a rectangle on the document by clicking and dragging →
                    text is auto-extracted. Or use <strong>Auto-Extract All</strong> for one-shot extraction.
                </div>
                """)

                # Document image + coordinate capture
                doc_image = gr.Image(
                    label="",
                    show_label=False,
                    interactive=True,       # allows selection
                    type="pil",
                    height=780,
                    elem_classes=["doc-viewer"],
                )

                # Coordinate inputs (hidden) — populated by JS from image click
                with gr.Row(visible=False):
                    sel_x1 = gr.Number(value=0, label="x1")
                    sel_y1 = gr.Number(value=0, label="y1")
                    sel_x2 = gr.Number(value=0, label="x2")
                    sel_y2 = gr.Number(value=0, label="y2")
                    do_extract_btn = gr.Button("Extract Region")

        # ── Event wiring ──────────────────────────────────────────────────────

        # Load document
        def on_upload(file_obj):
            pages, msg = load_document(file_obj)
            if not pages:
                return pages, 0, msg, None, '<div class="status-err">'+msg+'</div>'
            img = pages[0]
            status = f'<div class="status-ok">{msg}</div>'
            page_lbl = f'<div style="text-align:center;color:#6b6b78;font-size:12px;font-family:monospace;padding:8px">Page 1 / {len(pages)}</div>'
            return pages, 0, page_lbl, img, status

        upload_btn.change(
            on_upload,
            inputs=[upload_btn],
            outputs=[pages_state, page_idx_state, page_label, doc_image, status_box]
        )

        # Page navigation
        def nav_page(pages, idx, delta):
            if not pages:
                return idx, None, '<div style="text-align:center;color:#6b6b78;font-size:12px;font-family:monospace;padding:8px">—</div>'
            new_idx = max(0, min(len(pages) - 1, idx + delta))
            lbl = f'<div style="text-align:center;color:#6b6b78;font-size:12px;font-family:monospace;padding:8px">Page {new_idx+1} / {len(pages)}</div>'
            return new_idx, pages[new_idx], lbl

        prev_btn.click(lambda p, i: nav_page(p, i, -1), [pages_state, page_idx_state],
                       [page_idx_state, doc_image, page_label])
        next_btn.click(lambda p, i: nav_page(p, i, +1), [pages_state, page_idx_state],
                       [page_idx_state, doc_image, page_label])

        # Preset → fields
        def on_preset_change(preset):
            fields = PRESETS.get(preset, [])
            field_text = "\n".join(fields)
            choices = fields if fields else []
            val = fields[0] if fields else None
            return field_text, gr.update(choices=choices, value=val)

        preset_dd.change(on_preset_change, [preset_dd], [fields_box, active_field_dd])

        # Fields box → update active field dropdown
        def on_fields_change(text):
            fields = [f.strip() for f in text.splitlines() if f.strip()]
            return gr.update(choices=fields, value=fields[0] if fields else None)

        fields_box.change(on_fields_change, [fields_box], [active_field_dd])

        # Auto-extract all fields
        def on_auto_extract(pages, idx, fields_text, extracted, api_key):
            if not pages:
                return extracted, [], '<div class="status-err">❌ Upload a document first.</div>'
            fields = [f.strip() for f in fields_text.splitlines() if f.strip()]
            img = pages[idx]
            try:
                result = auto_extract_fields(img, fields, api_key)
                new_extracted = {**extracted, **result}
                rows = build_fields_df(fields, new_extracted)
                status = f'<div class="status-ok">✅ Extracted {len(result)} field(s) successfully.</div>'
                return new_extracted, rows, status
            except Exception as e:
                return extracted, build_fields_df(fields, extracted), f'<div class="status-err">❌ {e}</div>'

        extract_btn.click(
            on_auto_extract,
            inputs=[pages_state, page_idx_state, fields_box, extracted_state, api_key_input],
            outputs=[extracted_state, results_df, status_box]
        )

        # Region extract (triggered by hidden button)
        def on_region_extract(pages, idx, x1, y1, x2, y2, field, fields_text, extracted, boxes, api_key):
            if not pages or not field:
                return extracted, [], boxes, pages[idx] if pages else None, \
                       '<div class="status-err">❌ Select a field and upload a document first.</div>'
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                return extracted, build_fields_df(
                    [f.strip() for f in fields_text.splitlines() if f.strip()], extracted
                ), boxes, pages[idx], '<div class="status-err">❌ Selection too small.</div>'

            img = pages[idx]
            iw, ih = img.size
            # Gradio image coords are already in pixel space
            bx1, by1, bx2, by2 = int(min(x1,x2)), int(min(y1,y2)), int(max(x1,x2)), int(max(y1,y2))
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(iw, bx2), min(ih, by2)
            try:
                text = extract_region(img, bx1, by1, bx2, by2, field, api_key)
                new_extracted = {**extracted, field: text}
                new_boxes = boxes + [{"x1": bx1, "y1": by1, "x2": bx2, "y2": by2}]
                annotated = annotate_page(img, new_boxes)
                fields = [f.strip() for f in fields_text.splitlines() if f.strip()]
                rows = build_fields_df(fields, new_extracted)
                status = f'<div class="status-ok">✅ {field} → "{text[:60]}{"…" if len(text)>60 else ""}"</div>'
                return new_extracted, rows, new_boxes, annotated, status
            except Exception as e:
                return extracted, build_fields_df(
                    [f.strip() for f in fields_text.splitlines() if f.strip()], extracted
                ), boxes, pages[idx], f'<div class="status-err">❌ {e}</div>'

        do_extract_btn.click(
            on_region_extract,
            inputs=[pages_state, page_idx_state, sel_x1, sel_y1, sel_x2, sel_y2,
                    active_field_dd, fields_box, extracted_state, boxes_state, api_key_input],
            outputs=[extracted_state, results_df, boxes_state, doc_image, status_box]
        )

        # Clear
        def on_clear(fields_text):
            fields = [f.strip() for f in fields_text.splitlines() if f.strip()]
            return {}, build_fields_df(fields, {}), [], \
                   '<div class="status-ok">🗑 Cleared.</div>'

        clear_btn.click(on_clear, [fields_box],
                        [extracted_state, results_df, boxes_state, status_box])

        # Export JSON
        def on_export_json(extracted):
            path = "/tmp/extracted.json"
            with open(path, "w") as f:
                json.dump(extracted, f, indent=2)
            return gr.update(value=path, visible=True)

        export_json_btn.click(on_export_json, [extracted_state], [json_output])

        # Export CSV
        def on_export_csv(extracted):
            path = "/tmp/extracted.csv"
            lines = ["Field,Value"] + [f'"{k}","{v}"' for k, v in extracted.items()]
            with open(path, "w") as f:
                f.write("\n".join(lines))
            return gr.update(value=path, visible=True)

        export_csv_btn.click(on_export_csv, [extracted_state], [csv_output])

        # Inject JS for drag-select on image
        gr.HTML("""
        <script>
        (function() {
            let startX, startY, dragging = false;
            let overlay;

            function getDocImg() {
                // Find the image inside .doc-viewer
                return document.querySelector('.doc-viewer img');
            }

            function ensureOverlay(img) {
                if (overlay && overlay.parentNode) return overlay;
                overlay = document.createElement('div');
                overlay.id = 'sel-overlay';
                overlay.style.cssText = `
                    position:absolute;border:2px solid #F59E0B;
                    background:rgba(245,158,11,0.12);pointer-events:none;display:none;
                `;
                img.parentElement.style.position = 'relative';
                img.parentElement.appendChild(overlay);
                return overlay;
            }

            function imgCoords(img, clientX, clientY) {
                const r = img.getBoundingClientRect();
                const sx = img.naturalWidth  / r.width;
                const sy = img.naturalHeight / r.height;
                return {
                    x: Math.round((clientX - r.left) * sx),
                    y: Math.round((clientY - r.top)  * sy),
                    rx: clientX - r.left,
                    ry: clientY - r.top,
                };
            }

            function setHidden(label, val) {
                const inputs = document.querySelectorAll('input[type=number]');
                const labels = ['x1','y1','x2','y2'];
                const idx = labels.indexOf(label);
                if (idx >= 0 && inputs[idx]) {
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
                    nativeInputValueSetter.call(inputs[idx], val);
                    inputs[idx].dispatchEvent(new Event('input', {bubbles:true}));
                }
            }

            document.addEventListener('mousedown', function(e) {
                const img = getDocImg();
                if (!img || !img.contains(e.target) && e.target !== img) return;
                const c = imgCoords(img, e.clientX, e.clientY);
                startX = c.x; startY = c.y;
                dragging = true;
                const ov = ensureOverlay(img);
                const r  = img.getBoundingClientRect();
                ov.style.left   = (e.clientX - r.left) + 'px';
                ov.style.top    = (e.clientY - r.top)  + 'px';
                ov.style.width  = '0px';
                ov.style.height = '0px';
                ov.style.display = 'block';
                e.preventDefault();
            });

            document.addEventListener('mousemove', function(e) {
                if (!dragging) return;
                const img = getDocImg();
                if (!img) return;
                const r = img.getBoundingClientRect();
                const ov = ensureOverlay(img);
                const cx = e.clientX - r.left, cy = e.clientY - r.top;
                const ox = Math.min(startX * r.width / img.naturalWidth, cx);
                const oy = Math.min(startY * r.height / img.naturalHeight, cy);
                const ow = Math.abs(cx - startX * r.width / img.naturalWidth);
                const oh = Math.abs(cy - startY * r.height / img.naturalHeight);
                ov.style.left   = ox + 'px';
                ov.style.top    = oy + 'px';
                ov.style.width  = ow + 'px';
                ov.style.height = oh + 'px';
            });

            document.addEventListener('mouseup', function(e) {
                if (!dragging) return;
                dragging = false;
                const img = getDocImg();
                if (!img) return;
                const c = imgCoords(img, e.clientX, e.clientY);
                setHidden('x1', Math.min(startX, c.x));
                setHidden('y1', Math.min(startY, c.y));
                setHidden('x2', Math.max(startX, c.x));
                setHidden('y2', Math.max(startY, c.y));
                // Click the hidden extract button
                setTimeout(() => {
                    const btns = document.querySelectorAll('button');
                    for (const b of btns) {
                        if (b.textContent.trim() === 'Extract Region') { b.click(); break; }
                    }
                }, 80);
                if (overlay) overlay.style.display = 'none';
            });
        })();
        </script>
        """)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
# Build at module level so HF Spaces can find the `demo` object directly
demo = build_ui()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
