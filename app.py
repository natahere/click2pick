import streamlit as st
import json, io, os, re, tempfile
import fitz
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import pytesseract
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="DocExtract", page_icon="🔍", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap');
html,body,[class*="css"]{font-family:'Lato',sans-serif!important}
.stApp{background:#0c0c0f;color:#e2dfd8}
.app-title{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:#fff;letter-spacing:-.5px}
.app-title span{color:#F59E0B}
.app-sub{font-size:12px;color:#6b6b78;margin-top:3px}
.panel-label{font-family:'JetBrains Mono',monospace;font-size:10px;color:#6b6b78;
    letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.infobox{background:#13130f;border-left:3px solid #F59E0B;border-radius:6px;
    padding:9px 13px;font-size:12px;color:#a89060;margin-bottom:10px}
.step-badge{display:inline-block;background:#F59E0B;color:#0c0c0f;border-radius:50%;
    width:22px;height:22px;text-align:center;line-height:22px;font-weight:800;
    font-size:12px;margin-right:6px;font-family:'Syne',sans-serif}
.stButton>button{background:#F59E0B!important;color:#0c0c0f!important;border:none!important;
    border-radius:8px!important;font-weight:700!important}
.stButton>button:hover{background:#d97706!important}
.stSelectbox>div>div,.stTextInput>div>div>input,.stTextArea>div>div>textarea{
    background:#1c1c21!important;border:1px solid #26262d!important;
    border-radius:8px!important;color:#e2dfd8!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:0!important}
hr{border-color:#1e1e22!important}
</style>
""", unsafe_allow_html=True)

# ── OCR ───────────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    w, h = gray.size
    if w < 400:
        scale = 400 / w
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return gray

def run_ocr(img: Image.Image) -> str:
    processed = preprocess(img)
    r6  = pytesseract.image_to_string(processed, config="--psm 6  --oem 3").strip()
    r11 = pytesseract.image_to_string(processed, config="--psm 11 --oem 3").strip()
    return r6 if len(r6) >= len(r11) else r11

def pdf_to_pages(path: str):
    doc = fitz.open(path)
    pages, texts = [], []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        texts.append(page.get_text())
    return pages, texts

def smart_match(full_text: str, field: str) -> str:
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    fkey  = re.sub(r'\s+', '', field.lower())
    for i, line in enumerate(lines):
        if fkey in re.sub(r'\s+', '', line.lower()):
            parts = re.split(r'[:\t]\s*', line, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
            if i+1 < len(lines) and ':' not in lines[i+1] and len(lines[i+1]) < 80:
                return lines[i+1]
    words = [w for w in field.lower().split() if len(w) > 3]
    if words:
        for line in lines:
            if all(w in line.lower() for w in words):
                parts = re.split(r'[:\t]\s*', line, maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip()
    return ""

def auto_extract(img: Image.Image, fields: list, pdf_text: str = "") -> dict:
    text = pdf_text.strip() if pdf_text.strip() else run_ocr(img)
    if not fields:
        result = {}
        for line in text.splitlines():
            if ':' in line:
                k, _, v = line.partition(':')
                k, v = k.strip(), v.strip()
                if k and v and len(k) < 50:
                    result[k] = v
        return result
    return {f: smart_match(text, f) for f in fields}

def render_image(img: Image.Image, boxes: list,
                 pt1=None, pt2=None, disp_w=700) -> Image.Image:
    """Render image with completed boxes + current in-progress selection."""
    iw, ih  = img.size
    scale   = disp_w / iw
    disp_h  = int(ih * scale)
    display = img.resize((disp_w, disp_h), Image.LANCZOS).copy()
    draw    = ImageDraw.Draw(display, "RGBA")

    # Completed extractions in green
    for b in boxes:
        bx1 = int(b["x1"] * scale); by1 = int(b["y1"] * scale)
        bx2 = int(b["x2"] * scale); by2 = int(b["y2"] * scale)
        draw.rectangle([bx1,by1,bx2,by2], outline="#34d399", width=2,
                       fill=(52,211,153,30))

    # Point 1 marker
    if pt1:
        px, py = int(pt1[0] * scale), int(pt1[1] * scale)
        r = 7
        draw.ellipse([px-r, py-r, px+r, py+r], fill="#F59E0B", outline="#fff", width=2)
        draw.text((px+10, py-8), "1", fill="#F59E0B")

    # In-progress selection rectangle
    if pt1 and pt2:
        rx1 = int(min(pt1[0], pt2[0]) * scale)
        ry1 = int(min(pt1[1], pt2[1]) * scale)
        rx2 = int(max(pt1[0], pt2[0]) * scale)
        ry2 = int(max(pt1[1], pt2[1]) * scale)
        draw.rectangle([rx1,ry1,rx2,ry2], outline="#F59E0B", width=3,
                       fill=(245,158,11,45))
        draw.ellipse([int(pt2[0]*scale)-7, int(pt2[1]*scale)-7,
                      int(pt2[0]*scale)+7, int(pt2[1]*scale)+7],
                     fill="#F59E0B", outline="#fff", width=2)
        draw.text((int(pt2[0]*scale)+10, int(pt2[1]*scale)-8), "2", fill="#F59E0B")

    return display

# ── Presets ───────────────────────────────────────────────────────────────────
PRESETS = {
    "📄 Invoice":       ["Vendor Name","Invoice Number","Invoice Date","Due Date",
                         "Subtotal","Tax Amount","Total Amount","Payment Terms","PO Number"],
    "🪪 ID / Passport": ["Full Name","Date of Birth","Gender","Document Number",
                         "Nationality","Issue Date","Expiry Date","Place of Birth"],
    "🧾 Receipt":       ["Store Name","Store Address","Date","Time",
                         "Items","Subtotal","Tax","Total","Payment Method"],
    "📝 Contract":      ["Party A","Party B","Contract Date","Effective Date",
                         "Contract Value","Duration","Jurisdiction"],
    "🔑 Generic Key-Value": [],
    "✏️ Custom":        [],
}

DISP_W = 700   # display width in pixels

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = dict(pages=[], pdf_texts=[], extracted={}, boxes=[],
                page_idx=0, active_field=None,
                fields=list(PRESETS["📄 Invoice"]),
                pt1=None, pt2=None)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 14px;border-bottom:1px solid #26262d;margin-bottom:20px">
  <div class="app-title">Doc<span>Extract</span></div>
  <div class="app-sub">Upload · Click two corners on the image · Extract with OCR</div>
</div>""", unsafe_allow_html=True)
st.markdown('<span style="font-family:monospace;font-size:11px;color:#34d399">● TESSERACT active</span>',
            unsafe_allow_html=True)

left, right = st.columns([1, 1.9], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT
# ══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="panel-label">📋 Template</div>', unsafe_allow_html=True)
    preset = st.selectbox("Template", list(PRESETS.keys()), label_visibility="collapsed")
    if st.button("Load Template"):
        st.session_state.fields    = list(PRESETS[preset])
        st.session_state.extracted = {}
        st.rerun()

    with st.expander("✏️ Edit Fields"):
        ft = st.text_area("One per line", value="\n".join(st.session_state.fields),
                           height=140, label_visibility="collapsed")
        if st.button("Update Fields"):
            st.session_state.fields = [f.strip() for f in ft.splitlines() if f.strip()]
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">🎯 Target Field</div>', unsafe_allow_html=True)
    opts    = ["— select —"] + st.session_state.fields
    cur_idx = (st.session_state.fields.index(st.session_state.active_field) + 1
               if st.session_state.active_field in st.session_state.fields else 0)
    active  = st.selectbox("Field", opts, index=cur_idx, label_visibility="collapsed")
    st.session_state.active_field = None if active == "— select —" else active

    if st.session_state.pages:
        if st.button("⚡ Auto-Extract All", use_container_width=True):
            with st.spinner("Running OCR on full page…"):
                try:
                    idx      = st.session_state.page_idx
                    pdf_text = st.session_state.pdf_texts[idx] if st.session_state.pdf_texts else ""
                    result   = auto_extract(st.session_state.pages[idx],
                                            st.session_state.fields, pdf_text)
                    st.session_state.extracted.update(result)
                    st.success(f"✅ Extracted {len(result)} fields")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">📊 Extracted Data</div>', unsafe_allow_html=True)
    for field in st.session_state.fields:
        val     = st.session_state.extracted.get(field, "")
        is_act  = field == st.session_state.active_field
        label   = f"**→ {field}**" if is_act else field
        new_val = st.text_input(label, value=val, key=f"inp_{field}",
                                 placeholder="not extracted yet")
        if new_val != val:
            st.session_state.extracted[field] = new_val

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ JSON",
            json.dumps(st.session_state.extracted, indent=2),
            "extracted.json", "application/json", use_container_width=True)
    with c2:
        csv_str = "\n".join(["Field,Value"] +
                  [f'"{k}","{v}"' for k,v in st.session_state.extracted.items()])
        st.download_button("⬇ CSV", csv_str, "extracted.csv", "text/csv", use_container_width=True)

    if st.button("🗑 Clear All", use_container_width=True):
        st.session_state.extracted = {}
        st.session_state.boxes     = []
        st.session_state.pt1       = None
        st.session_state.pt2       = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT
# ══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="panel-label">📄 Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=["pdf","png","jpg","jpeg","webp","tiff","bmp"],
                                  label_visibility="collapsed")
    if uploaded:
        data = uploaded.read()
        ext  = uploaded.name.rsplit(".",1)[-1].lower()
        if ext == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data); tmp_path = tmp.name
            pages, texts = pdf_to_pages(tmp_path)
            os.unlink(tmp_path)
            st.session_state.pages     = pages
            st.session_state.pdf_texts = texts
        else:
            st.session_state.pages     = [Image.open(io.BytesIO(data)).convert("RGB")]
            st.session_state.pdf_texts = []
        st.session_state.page_idx = 0
        st.session_state.boxes    = []
        st.session_state.pt1      = None
        st.session_state.pt2      = None

    if not st.session_state.pages:
        st.markdown("""<div style="border:2px dashed #26262d;border-radius:12px;
            padding:60px;text-align:center">
          <div style="font-size:48px">📄</div>
          <div style="color:#6b6b78;margin-top:12px">Upload a PDF or image to get started</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Page nav
    n = len(st.session_state.pages)
    if n > 1:
        c1, c2, c3 = st.columns([1,2,1])
        with c1:
            if st.button("◀ Prev") and st.session_state.page_idx > 0:
                st.session_state.page_idx -= 1
                st.session_state.boxes    = []
                st.session_state.pt1      = None
                st.session_state.pt2      = None
                st.rerun()
        with c2:
            st.markdown(f'<div style="text-align:center;color:#6b6b78;font-size:12px;'
                        f'font-family:monospace;padding:8px">'
                        f'Page {st.session_state.page_idx+1} / {n}</div>',
                        unsafe_allow_html=True)
        with c3:
            if st.button("Next ▶") and st.session_state.page_idx < n-1:
                st.session_state.page_idx += 1
                st.session_state.boxes    = []
                st.session_state.pt1      = None
                st.session_state.pt2      = None
                st.rerun()

    page_img = st.session_state.pages[st.session_state.page_idx]
    iw, ih   = page_img.size
    pt1      = st.session_state.pt1
    pt2      = st.session_state.pt2

    # ── Instruction bar ───────────────────────────────────────────────────────
    if not st.session_state.active_field:
        st.markdown('<div class="infobox">👈 Select a <strong>Target Field</strong> on the left first.</div>',
                    unsafe_allow_html=True)
    elif pt1 is None:
        st.markdown(f'<div class="infobox">'
                    f'<span class="step-badge">1</span>'
                    f'Click the <strong>top-left corner</strong> of the text region for '
                    f'<strong>{st.session_state.active_field}</strong></div>',
                    unsafe_allow_html=True)
    elif pt2 is None:
        st.markdown(f'<div class="infobox">'
                    f'<span class="step-badge">2</span>'
                    f'Now click the <strong>bottom-right corner</strong> of the region for '
                    f'<strong>{st.session_state.active_field}</strong></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="infobox">✅ Region selected — click '
                    '<strong>Extract</strong> below, or click the image to re-select.</div>',
                    unsafe_allow_html=True)

    # ── Render image with annotations ─────────────────────────────────────────
    display_img = render_image(page_img, st.session_state.boxes, pt1, pt2, DISP_W)

    # streamlit_image_coordinates returns {x, y} in DISPLAY pixel coords on click
    coords = streamlit_image_coordinates(display_img, key=f"img_{st.session_state.page_idx}")

    # ── Handle click ──────────────────────────────────────────────────────────
    if coords is not None:
        # Convert display coords → original image coords
        scale = iw / DISP_W
        orig_x = int(coords["x"] * scale)
        orig_y = int(coords["y"] * scale)

        if pt1 is None:
            # First click — set top-left
            st.session_state.pt1 = (orig_x, orig_y)
            st.rerun()
        elif pt2 is None:
            # Second click — set bottom-right
            st.session_state.pt2 = (orig_x, orig_y)
            st.rerun()
        else:
            # Third click — start over with new point 1
            st.session_state.pt1 = (orig_x, orig_y)
            st.session_state.pt2 = None
            st.rerun()

    # ── Crop preview + Extract button ─────────────────────────────────────────
    if pt1 and pt2:
        x1 = max(0,  min(pt1[0], pt2[0]) - 6)
        y1 = max(0,  min(pt1[1], pt2[1]) - 6)
        x2 = min(iw, max(pt1[0], pt2[0]) + 6)
        y2 = min(ih, max(pt1[1], pt2[1]) + 6)

        if x2 - x1 > 8 and y2 - y1 > 8:
            crop = page_img.crop((x1, y1, x2, y2))

            st.markdown('<div class="panel-label" style="margin-top:10px">🔎 Crop Preview</div>',
                        unsafe_allow_html=True)
            preview_w = min(crop.width * 2, DISP_W)
            st.image(crop, width=preview_w)

            btn_col, reset_col = st.columns([3, 1])
            with btn_col:
                btn_label = (f"🔍 Extract → \"{st.session_state.active_field}\""
                             if st.session_state.active_field else "🔍 Extract Text")
                if st.button(btn_label, use_container_width=True, type="primary"):
                    with st.spinner("Running OCR…"):
                        try:
                            text = run_ocr(crop)
                            if not text:
                                st.warning("No text detected — try selecting a wider region.")
                            else:
                                st.success(f"✅ `{text[:120]}{'…' if len(text)>120 else ''}`")
                                if st.session_state.active_field:
                                    st.session_state.extracted[st.session_state.active_field] = text
                                    st.session_state.boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
                                    fields = st.session_state.fields
                                    if st.session_state.active_field in fields:
                                        idx = fields.index(st.session_state.active_field)
                                        if idx + 1 < len(fields):
                                            st.session_state.active_field = fields[idx+1]
                                    st.session_state.pt1 = None
                                    st.session_state.pt2 = None
                                    st.rerun()
                        except Exception as e:
                            st.error(f"OCR error: {e}")
            with reset_col:
                if st.button("↺ Reset", use_container_width=True):
                    st.session_state.pt1 = None
                    st.session_state.pt2 = None
                    st.rerun()
