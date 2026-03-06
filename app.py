import streamlit as st
import anthropic
import base64
import json
import io
import fitz  # PyMuPDF
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocExtract",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0f0f11;
    color: #e8e6e1;
}

/* Header */
.main-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 0 8px 0;
    border-bottom: 1px solid #222;
    margin-bottom: 24px;
}
.logo-mark {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #f0a500, #e06c00);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.app-title {
    font-size: 22px;
    font-weight: 600;
    color: #f0f0f0;
    letter-spacing: -0.3px;
}
.app-subtitle {
    font-size: 13px;
    color: #666;
    margin-top: 2px;
}

/* Panel labels */
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: #555;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Field cards */
.field-card {
    background: #161618;
    border: 1px solid #242428;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.field-card:hover {
    border-color: #f0a500;
}
.field-label {
    font-size: 11px;
    font-weight: 500;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}
.field-value {
    font-size: 15px;
    color: #e8e6e1;
    min-height: 20px;
}
.field-empty {
    color: #3a3a3e;
    font-style: italic;
    font-size: 13px;
}

/* Upload zone */
.upload-zone {
    border: 2px dashed #2a2a2e;
    border-radius: 12px;
    padding: 40px;
    text-align: center;
    transition: border-color 0.2s;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a1a1e;
    border: 1px solid #2a2a2e;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #888;
    font-family: 'DM Mono', monospace;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #f0a500;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Buttons */
.stButton > button {
    background: #f0a500 !important;
    color: #0f0f11 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #e09500 !important;
    transform: translateY(-1px) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #161618 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 8px !important;
    color: #e8e6e1 !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: #161618 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 8px !important;
    color: #e8e6e1 !important;
}

/* Text area */
.stTextArea > div > div > textarea {
    background: #161618 !important;
    border: 1px solid #2a2a2e !important;
    border-radius: 8px !important;
    color: #e8e6e1 !important;
    font-family: 'DM Mono', monospace !important;
}

/* Divider */
hr { border-color: #1e1e22 !important; }

/* Instruction box */
.instruction-box {
    background: #13130f;
    border: 1px solid #2a2600;
    border-left: 3px solid #f0a500;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #a89060;
    margin-bottom: 16px;
}

/* Page nav */
.page-nav {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    align-items: center;
}
.page-info {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #555;
}

/* JSON output */
.json-output {
    background: #0a0a0c;
    border: 1px solid #1e1e22;
    border-radius: 8px;
    padding: 16px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #a0c080;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for clarity
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def image_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode()


def crop_region(img: Image.Image, rect: dict, canvas_w: int, canvas_h: int) -> Image.Image:
    """Map canvas coordinates → image coordinates and crop."""
    iw, ih = img.size
    sx = iw / canvas_w
    sy = ih / canvas_h
    x1 = int(rect["left"] * sx)
    y1 = int(rect["top"] * sy)
    x2 = int((rect["left"] + rect["width"]) * sx)
    y2 = int((rect["top"] + rect["height"]) * sy)
    x1, x2 = max(0, x1), min(iw, x2)
    y1, y2 = max(0, y1), min(ih, y2)
    return img.crop((x1, y1, x2, y2))


def extract_text_claude(cropped_img: Image.Image, field_name: str, api_key: str) -> str:
    """Send cropped region to Claude Vision and extract text."""
    client = anthropic.Anthropic(api_key=api_key)
    b64 = image_to_b64(cropped_img)
    prompt = (
        f"You are a precise OCR engine. Extract ONLY the text visible in this image region. "
        f"This text will be used for the field: '{field_name}'. "
        f"Return ONLY the extracted text with no explanation, no quotes, no formatting. "
        f"If no text is visible, return an empty string."
    )
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt},
            ]
        }]
    )
    return response.content[0].text.strip()


def extract_all_fields_claude(img: Image.Image, fields: list[str], api_key: str) -> dict:
    """Send full page to Claude and extract all fields at once."""
    client = anthropic.Anthropic(api_key=api_key)
    b64 = image_to_b64(img)
    fields_json = json.dumps(fields)
    prompt = (
        f"You are a precise document data extractor. "
        f"Extract the following fields from this document image: {fields_json}. "
        f"Return ONLY a valid JSON object with field names as keys and extracted values as strings. "
        f"If a field is not found, use an empty string. No explanation, no markdown."
    )
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt},
            ]
        }]
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ── Preset field templates ────────────────────────────────────────────────────
PRESETS = {
    "Invoice": ["Vendor Name", "Invoice Number", "Invoice Date", "Due Date", "Total Amount", "Tax Amount", "Subtotal", "Payment Terms"],
    "ID / Passport": ["Full Name", "Date of Birth", "Document Number", "Nationality", "Issue Date", "Expiry Date", "Place of Birth"],
    "Receipt": ["Store Name", "Date", "Total", "Tax", "Payment Method", "Items"],
    "Contract": ["Party A", "Party B", "Contract Date", "Effective Date", "Contract Value", "Jurisdiction"],
    "Custom": [],
}

CANVAS_W = 700

# ── Session state ─────────────────────────────────────────────────────────────
if "fields" not in st.session_state:
    st.session_state.fields = list(PRESETS["Invoice"])
if "extracted" not in st.session_state:
    st.session_state.extracted = {}
if "pages" not in st.session_state:
    st.session_state.pages = []
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "active_field" not in st.session_state:
    st.session_state.active_field = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="logo-mark">🔍</div>
    <div>
        <div class="app-title">DocExtract</div>
        <div class="app-subtitle">Click regions on your document to extract data</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Layout: Left panel (fields) | Right panel (document) ─────────────────────
left, right = st.columns([1, 1.8], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Configuration & Extracted Fields
# ══════════════════════════════════════════════════════════════════════════════
with left:
    # API Key
    st.markdown('<div class="panel-label">⚙️ Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API Key", type="password",
                             value=st.session_state.api_key,
                             placeholder="sk-ant-...",
                             help="Get your key at console.anthropic.com")
    st.session_state.api_key = api_key

    st.markdown("<hr>", unsafe_allow_html=True)

    # Template selector
    st.markdown('<div class="panel-label">📋 Field Template</div>', unsafe_allow_html=True)
    preset_name = st.selectbox("Template", list(PRESETS.keys()), label_visibility="collapsed")

    if preset_name != "Custom":
        if st.button("Load Template"):
            st.session_state.fields = list(PRESETS[preset_name])
            st.session_state.extracted = {}
            st.rerun()

    # Custom fields editor
    with st.expander("✏️ Edit Fields", expanded=(preset_name == "Custom")):
        fields_text = st.text_area(
            "One field per line",
            value="\n".join(st.session_state.fields),
            height=160,
            label_visibility="collapsed",
        )
        if st.button("Update Fields"):
            st.session_state.fields = [f.strip() for f in fields_text.splitlines() if f.strip()]
            st.session_state.extracted = {f: st.session_state.extracted.get(f, "") for f in st.session_state.fields}
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Active field selector
    st.markdown('<div class="panel-label">🎯 Active Field (click-to-fill)</div>', unsafe_allow_html=True)
    active = st.selectbox(
        "Select field to fill by clicking the document",
        ["— select —"] + st.session_state.fields,
        label_visibility="collapsed"
    )
    st.session_state.active_field = None if active == "— select —" else active

    # Auto-extract all button
    if st.session_state.pages:
        if st.button("⚡ Auto-Extract All Fields", use_container_width=True):
            if not st.session_state.api_key:
                st.error("Please enter your Anthropic API Key.")
            else:
                with st.spinner("Extracting all fields with Claude Vision..."):
                    try:
                        result = extract_all_fields_claude(
                            st.session_state.pages[st.session_state.page_idx],
                            st.session_state.fields,
                            st.session_state.api_key
                        )
                        for k, v in result.items():
                            if k in st.session_state.fields:
                                st.session_state.extracted[k] = v
                        st.success("Extraction complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Extracted fields display + editing
    st.markdown('<div class="panel-label">📊 Extracted Data</div>', unsafe_allow_html=True)

    changed = False
    for field in st.session_state.fields:
        val = st.session_state.extracted.get(field, "")
        is_active = field == st.session_state.active_field

        border = "#f0a500" if is_active else "#242428"
        st.markdown(f"""
        <div class="field-card" style="border-color:{border};">
            <div class="field-label">{'→ ' if is_active else ''}{field}</div>
        </div>
        """, unsafe_allow_html=True)

        new_val = st.text_input(f"_{field}", value=val, label_visibility="collapsed", key=f"inp_{field}")
        if new_val != val:
            st.session_state.extracted[field] = new_val
            changed = True

    if changed:
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Export
    st.markdown('<div class="panel-label">💾 Export</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        json_str = json.dumps(st.session_state.extracted, indent=2)
        st.download_button("⬇ JSON", json_str, "extracted.json", "application/json", use_container_width=True)
    with col_b:
        csv_lines = ["Field,Value"] + [f'"{k}","{v}"' for k, v in st.session_state.extracted.items()]
        st.download_button("⬇ CSV", "\n".join(csv_lines), "extracted.csv", "text/csv", use_container_width=True)

    if st.session_state.extracted:
        st.markdown(f'<div class="json-output">{json.dumps(st.session_state.extracted, indent=2)}</div>',
                    unsafe_allow_html=True)

    if st.button("🗑 Clear All", use_container_width=True):
        st.session_state.extracted = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Document Viewer + Canvas
# ══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="panel-label">📄 Document Viewer</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload document",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded:
        file_bytes = uploaded.read()
        ext = uploaded.name.split(".")[-1].lower()

        if ext == "pdf":
            with st.spinner("Rendering PDF..."):
                st.session_state.pages = pdf_to_images(file_bytes)
        else:
            st.session_state.pages = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

        st.session_state.page_idx = min(st.session_state.page_idx, len(st.session_state.pages) - 1)

    if st.session_state.pages:
        n_pages = len(st.session_state.pages)

        # Page navigation
        if n_pages > 1:
            st.markdown('<div class="page-nav">', unsafe_allow_html=True)
            nav_l, nav_info, nav_r = st.columns([1, 2, 1])
            with nav_l:
                if st.button("◀ Prev") and st.session_state.page_idx > 0:
                    st.session_state.page_idx -= 1
                    st.rerun()
            with nav_info:
                st.markdown(f'<div class="page-info" style="text-align:center">Page {st.session_state.page_idx+1} of {n_pages}</div>', unsafe_allow_html=True)
            with nav_r:
                if st.button("Next ▶") and st.session_state.page_idx < n_pages - 1:
                    st.session_state.page_idx += 1
                    st.rerun()

        current_page = st.session_state.pages[st.session_state.page_idx]
        iw, ih = current_page.size
        canvas_h = int(ih * CANVAS_W / iw)

        # Instruction
        if st.session_state.active_field:
            st.markdown(f"""
            <div class="instruction-box">
                🖱️ Draw a rectangle around the text for <strong>{st.session_state.active_field}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="instruction-box">
                👈 Select a field on the left, then draw a box around the matching text in the document.
            </div>
            """, unsafe_allow_html=True)

        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(240, 165, 0, 0.15)",
            stroke_width=2,
            stroke_color="#f0a500",
            background_image=current_page,
            update_streamlit=True,
            height=canvas_h,
            width=CANVAS_W,
            drawing_mode="rect",
            key=f"canvas_{st.session_state.page_idx}",
        )

        # Process drawn rectangle
        if (
            canvas_result.json_data
            and canvas_result.json_data.get("objects")
            and st.session_state.active_field
        ):
            obj = canvas_result.json_data["objects"][-1]
            if obj.get("width", 0) > 5 and obj.get("height", 0) > 5:
                if not st.session_state.api_key:
                    st.error("Enter your Anthropic API Key on the left to extract text.")
                else:
                    with st.spinner(f"Extracting '{st.session_state.active_field}'..."):
                        try:
                            cropped = crop_region(current_page, obj, CANVAS_W, canvas_h)
                            text = extract_text_claude(cropped, st.session_state.active_field, st.session_state.api_key)
                            st.session_state.extracted[st.session_state.active_field] = text

                            # Move to next field automatically
                            fields = st.session_state.fields
                            idx = fields.index(st.session_state.active_field)
                            if idx + 1 < len(fields):
                                st.session_state.active_field = fields[idx + 1]

                            st.success(f"✓ Extracted: {text[:60]}{'...' if len(text)>60 else ''}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Extraction error: {e}")

    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:48px;margin-bottom:16px">📄</div>
            <div style="color:#555;font-size:15px">Upload a PDF or image to get started</div>
            <div style="color:#3a3a3e;font-size:13px;margin-top:8px">Supports PDF, JPG, PNG</div>
        </div>
        """, unsafe_allow_html=True)
