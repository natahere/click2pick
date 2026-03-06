import streamlit as st
import json, io, os, re, tempfile, base64
import fitz
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import streamlit.components.v1 as components

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
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:0!important}
hr{border-color:#1e1e22!important}
</style>
""", unsafe_allow_html=True)

# ── OCR ───────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading OCR engine…")
def load_ocr():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return "tesseract", pytesseract
    except Exception:
        pass
    try:
        import easyocr
        return "easyocr", easyocr.Reader(['en'], gpu=False, verbose=False)
    except Exception:
        pass
    return "pymupdf", None

OCR_ENGINE, OCR_OBJ = load_ocr()

def preprocess(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    w, h = gray.size
    if w < 300:
        scale = 300 / w
        gray = gray.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return gray

def run_ocr(img: Image.Image) -> str:
    processed = preprocess(img)
    if OCR_ENGINE == "tesseract":
        import pytesseract
        r6  = pytesseract.image_to_string(processed, config="--psm 6  --oem 3").strip()
        r11 = pytesseract.image_to_string(processed, config="--psm 11 --oem 3").strip()
        return r6 if len(r6) >= len(r11) else r11
    elif OCR_ENGINE == "easyocr":
        import numpy as np
        arr = np.array(processed.convert("RGB"))
        return " ".join(OCR_OBJ.readtext(arr, detail=0, paragraph=True)).strip()
    return ""

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

def annotate(img: Image.Image, boxes: list) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for b in boxes:
        draw.rectangle([b["x1"],b["y1"],b["x2"],b["y2"]],
                       outline="#F59E0B", width=3, fill=(245,158,11,35))
    return out

def pil_to_b64(img: Image.Image, quality=88) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
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
                fields=list(PRESETS["📄 Invoice"]),
                click1=None, click2=None, crop_img=None)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 14px;border-bottom:1px solid #26262d;margin-bottom:20px">
  <div class="app-title">Doc<span>Extract</span></div>
  <div class="app-sub">Upload · Click two corners · Extract with OCR — no API key needed</div>
</div>""", unsafe_allow_html=True)

ec = {"tesseract":"#34d399","easyocr":"#818cf8","pymupdf":"#F59E0B"}.get(OCR_ENGINE,"#aaa")
st.markdown(f'<span style="font-family:monospace;font-size:11px;color:{ec}">● {OCR_ENGINE.upper()} active</span>',
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
        val       = st.session_state.extracted.get(field, "")
        is_active = field == st.session_state.active_field
        label     = f"**→ {field}**" if is_active else field
        new_val   = st.text_input(label, value=val, key=f"inp_{field}",
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
        st.session_state.click1    = None
        st.session_state.click2    = None
        st.session_state.crop_img  = None
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
        st.session_state.click1    = None
        st.session_state.click2    = None
        st.session_state.crop_img  = None

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
                st.session_state.click1   = None
                st.session_state.click2   = None
                st.session_state.crop_img = None
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
                st.session_state.click1   = None
                st.session_state.click2   = None
                st.session_state.crop_img = None
                st.rerun()

    page_img    = st.session_state.pages[st.session_state.page_idx]
    iw, ih      = page_img.size

    # ── Instruction ───────────────────────────────────────────────────────────
    c1_state = st.session_state.click1
    c2_state = st.session_state.click2

    if not st.session_state.active_field:
        st.markdown('<div class="infobox">👈 First select a <strong>Target Field</strong> on the left.</div>',
                    unsafe_allow_html=True)
    elif c1_state is None:
        st.markdown(f'<div class="infobox">🖱 <strong>Click #1</strong> — click the '
                    f'<strong>top-left corner</strong> of the region containing '
                    f'<strong>{st.session_state.active_field}</strong></div>',
                    unsafe_allow_html=True)
    elif c2_state is None:
        st.markdown(f'<div class="infobox">🖱 <strong>Click #2</strong> — click the '
                    f'<strong>bottom-right corner</strong> of the region containing '
                    f'<strong>{st.session_state.active_field}</strong></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="infobox">✅ Region selected — click <strong>Extract</strong> below, '
                    'or click on the image to re-select.</div>', unsafe_allow_html=True)

    # ── Interactive image rendered as HTML component ──────────────────────────
    # The component renders the image with click detection.
    # On each click it posts a message; we capture it via a hidden form trick
    # that stores click coords in st.session_state using st.query_params.
    disp_w  = 660
    disp_h  = int(ih * disp_w / iw)

    # Draw existing boxes + current selection on display image
    display_img = annotate(page_img, st.session_state.boxes)
    draw        = ImageDraw.Draw(display_img, "RGBA")

    # Draw click1 marker
    if c1_state:
        cx = int(c1_state[0] * disp_w / iw)
        cy = int(c1_state[1] * disp_h / ih)
        px1, py1 = c1_state
        draw.ellipse([px1-12, py1-12, px1+12, py1+12],
                     fill=(245,158,11,200), outline="#F59E0B", width=2)
        draw.text((px1+15, py1-8), "1", fill="#F59E0B")

    # Draw selection rectangle if both clicks exist
    if c1_state and c2_state:
        x1 = min(c1_state[0], c2_state[0])
        y1 = min(c1_state[1], c2_state[1])
        x2 = max(c1_state[0], c2_state[0])
        y2 = max(c1_state[1], c2_state[1])
        draw.rectangle([x1, y1, x2, y2],
                       outline="#F59E0B", width=3, fill=(245,158,11,40))

    b64 = pil_to_b64(display_img)

    # Render interactive image — clicks update query params → Streamlit reruns
    click_html = f"""<!DOCTYPE html><html><head><style>
    body{{margin:0;padding:0;background:#0c0c0f;overflow:hidden}}
    #wrap{{position:relative;display:inline-block;cursor:crosshair}}
    img{{display:block;border:1px solid #26262d;border-radius:10px;width:{disp_w}px}}
    #dot{{position:absolute;width:18px;height:18px;border-radius:50%;
          background:#F59E0B;border:2px solid #fff;
          transform:translate(-50%,-50%);display:none;pointer-events:none}}
    #msg{{font-family:monospace;font-size:11px;color:#6b6b78;margin-top:6px;
          padding:5px 8px;background:#141417;border-radius:5px}}
    </style></head><body>
    <div id="wrap">
      <img id="docimg" src="data:image/jpeg;base64,{b64}"/>
      <div id="dot"></div>
    </div>
    <div id="msg">Click on the document to set a corner point</div>
    <script>
    (function(){{
      const img   = document.getElementById('docimg');
      const dot   = document.getElementById('dot');
      const msg   = document.getElementById('msg');
      const scaleX = {iw} / {disp_w};
      const scaleY = {ih} / {disp_h};
      const clickNum = {1 if c1_state is None else (2 if c2_state is None else 1)};

      img.addEventListener('click', function(e) {{
        const r  = img.getBoundingClientRect();
        const rx = e.clientX - r.left;
        const ry = e.clientY - r.top;
        const px = Math.round(rx * scaleX);
        const py = Math.round(ry * scaleY);

        // Show dot
        dot.style.left = rx + 'px';
        dot.style.top  = ry + 'px';
        dot.style.display = 'block';
        msg.textContent = 'Click ' + clickNum + ' → (' + px + ', ' + py + ') — updating…';

        // Navigate to same URL with click params added
        const url = new URL(window.parent.location.href);
        url.searchParams.set('cx', px);
        url.searchParams.set('cy', py);
        url.searchParams.set('cn', clickNum);
        window.parent.location.href = url.toString();
      }});
    }})();
    </script></body></html>"""

    components.html(click_html, height=disp_h + 40, scrolling=False)

    # ── Read click from query params ──────────────────────────────────────────
    params = st.query_params
    if "cx" in params and "cy" in params and "cn" in params:
        try:
            cx = int(params["cx"])
            cy = int(params["cy"])
            cn = int(params["cn"])
            if cn == 1:
                st.session_state.click1 = (cx, cy)
                st.session_state.click2 = None
                st.session_state.crop_img = None
            else:
                st.session_state.click2 = (cx, cy)
            # Clear params and rerun cleanly
            st.query_params.clear()
            st.rerun()
        except Exception:
            st.query_params.clear()

    # ── Show crop preview + extract button ────────────────────────────────────
    if c1_state and c2_state:
        x1 = max(0, min(c1_state[0], c2_state[0]) - 6)
        y1 = max(0, min(c1_state[1], c2_state[1]) - 6)
        x2 = min(iw, max(c1_state[0], c2_state[0]) + 6)
        y2 = min(ih, max(c1_state[1], c2_state[1]) + 6)

        if x2 - x1 > 8 and y2 - y1 > 8:
            crop = page_img.crop((x1, y1, x2, y2))
            st.markdown('<div class="panel-label" style="margin-top:12px">🔎 Region Preview</div>',
                        unsafe_allow_html=True)
            st.image(crop, width=min(crop.width * 2, 600))

            btn_label = (f"🔍 Extract into \"{st.session_state.active_field}\""
                         if st.session_state.active_field else "🔍 Extract Text")

            if st.button(btn_label, use_container_width=True, type="primary"):
                with st.spinner("Running OCR…"):
                    try:
                        text = run_ocr(crop)
                        if not text:
                            st.warning("No text detected — try selecting a larger area "
                                       "or use ⚡ Auto-Extract All.")
                        else:
                            st.success(f"✅ `{text[:120]}{'…' if len(text)>120 else ''}`")
                            if st.session_state.active_field:
                                st.session_state.extracted[st.session_state.active_field] = text
                                st.session_state.boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
                                # Advance to next field
                                fields = st.session_state.fields
                                if st.session_state.active_field in fields:
                                    idx = fields.index(st.session_state.active_field)
                                    if idx + 1 < len(fields):
                                        st.session_state.active_field = fields[idx+1]
                                st.session_state.click1 = None
                                st.session_state.click2 = None
                                st.session_state.crop_img = None
                                st.rerun()
                    except Exception as e:
                        st.error(f"OCR error: {e}")

            col_r = st.columns([1,1])[1]
            with col_r:
                if st.button("↺ Re-select region", use_container_width=True):
                    st.session_state.click1   = None
                    st.session_state.click2   = None
                    st.session_state.crop_img = None
                    st.rerun()
