import streamlit as st
import json, io, os, re, tempfile, base64
import fitz
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import pytesseract

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
.panel-label{font-family:'JetBrains Mono',monospace;font-size:10px;color:#6b6b78;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.infobox{background:#13130f;border-left:3px solid #F59E0B;border-radius:6px;padding:9px 13px;font-size:12px;color:#a89060;margin-bottom:10px}
.stButton>button{background:#F59E0B!important;color:#0c0c0f!important;border:none!important;border-radius:8px!important;font-weight:700!important}
.stButton>button:hover{background:#d97706!important}
.stSelectbox>div>div,.stTextInput>div>div>input,.stTextArea>div>div>textarea{background:#1c1c21!important;border:1px solid #26262d!important;border-radius:8px!important;color:#e2dfd8!important}
.stSlider>div>div>div>div{background:#F59E0B!important}
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
        gray = gray.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
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

def draw_selection(img: Image.Image, boxes, x1, y1, x2, y2, show_sel=True) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    # Past boxes
    for b in boxes:
        draw.rectangle([b["x1"],b["y1"],b["x2"],b["y2"]],
                       outline="#34d399", width=2, fill=(52,211,153,25))
    # Current selection
    if show_sel and x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2],
                       outline="#F59E0B", width=3, fill=(245,158,11,40))
    return out

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()

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

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = dict(pages=[], pdf_texts=[], extracted={}, boxes=[],
                page_idx=0, active_field=None,
                fields=list(PRESETS["📄 Invoice"]))
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 14px;border-bottom:1px solid #26262d;margin-bottom:20px">
  <div class="app-title">Doc<span>Extract</span></div>
  <div class="app-sub">Upload · Use sliders to frame a region · Extract with OCR — no API key needed</div>
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
        st.session_state.page_idx  = 0
        st.session_state.boxes     = []

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
                st.session_state.boxes = []
                st.rerun()
        with c2:
            st.markdown(f'<div style="text-align:center;color:#6b6b78;font-size:12px;'
                        f'font-family:monospace;padding:8px">'
                        f'Page {st.session_state.page_idx+1} / {n}</div>',
                        unsafe_allow_html=True)
        with c3:
            if st.button("Next ▶") and st.session_state.page_idx < n-1:
                st.session_state.page_idx += 1
                st.session_state.boxes = []
                st.rerun()

    page_img = st.session_state.pages[st.session_state.page_idx]
    iw, ih   = page_img.size

    # ── Two-column layout: sliders left, preview right ────────────────────────
    sl_col, prev_col = st.columns([1, 2])

    with sl_col:
        st.markdown('<div class="panel-label">📐 Select Region</div>', unsafe_allow_html=True)
        st.caption("Move sliders to frame the text you want to extract")

        # Sliders as % of image dimensions for resolution-independence
        left_pct  = st.slider("Left %",   0, 100,  0, 1, key="sl_left")
        right_pct = st.slider("Right %",  0, 100, 100, 1, key="sl_right")
        top_pct   = st.slider("Top %",    0, 100,  0, 1, key="sl_top")
        bottom_pct= st.slider("Bottom %", 0, 100, 100, 1, key="sl_bottom")

        x1 = int(iw * left_pct   / 100)
        x2 = int(iw * right_pct  / 100)
        y1 = int(ih * top_pct    / 100)
        y2 = int(ih * bottom_pct / 100)

        has_sel = x2 > x1 + 10 and y2 > y1 + 10

        if has_sel:
            pad  = 4
            crop = page_img.crop((max(0,x1-pad), max(0,y1-pad),
                                   min(iw,x2+pad), min(ih,y2+pad)))

            st.markdown('<div class="panel-label" style="margin-top:14px">🔎 Region Preview</div>',
                        unsafe_allow_html=True)
            st.image(crop, use_container_width=True)

            btn_label = (f"🔍 Extract → \"{st.session_state.active_field}\""
                         if st.session_state.active_field else "🔍 Extract Text")

            if st.button(btn_label, use_container_width=True, type="primary"):
                with st.spinner("Running OCR…"):
                    try:
                        text = run_ocr(crop)
                        if not text:
                            st.warning("No text detected — adjust sliders to frame text more tightly.")
                        else:
                            st.success(f"✅ `{text[:100]}{'…' if len(text)>100 else ''}`")
                            if st.session_state.active_field:
                                st.session_state.extracted[st.session_state.active_field] = text
                                st.session_state.boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
                                fields = st.session_state.fields
                                if st.session_state.active_field in fields:
                                    idx = fields.index(st.session_state.active_field)
                                    if idx + 1 < len(fields):
                                        st.session_state.active_field = fields[idx+1]
                                st.rerun()
                    except Exception as e:
                        st.error(f"OCR error: {e}")

    with prev_col:
        st.markdown('<div class="panel-label">🗺 Document (with selection)</div>',
                    unsafe_allow_html=True)
        annotated = draw_selection(page_img, st.session_state.boxes,
                                    x1, y1, x2, y2, has_sel)
        st.image(annotated, use_container_width=True)
        if st.session_state.active_field:
            st.markdown(f'<div class="infobox" style="margin-top:8px">🎯 Framing region for: '
                        f'<strong>{st.session_state.active_field}</strong><br>'
                        f'Use sliders on the left to tighten the yellow box around the text, '
                        f'then click Extract.</div>', unsafe_allow_html=True)
