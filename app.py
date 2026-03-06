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
.field-row{background:#141417;border:1px solid #26262d;border-radius:8px;padding:10px 14px;margin-bottom:6px}
.field-name{font-family:'JetBrains Mono',monospace;font-size:10px;color:#6b6b78;text-transform:uppercase;letter-spacing:1px}
.field-val{font-size:13px;color:#e2dfd8;margin-top:3px}
.field-empty{color:#333;font-style:italic;font-size:11px}
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
    """Enhance image region for best OCR accuracy."""
    # Convert to grayscale
    gray = img.convert("L")
    # Increase contrast
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    # Sharpen
    gray = gray.filter(ImageFilter.SHARPEN)
    # Scale up small regions — Tesseract works best at 300 DPI equivalent
    w, h = gray.size
    if w < 300:
        scale = 300 / w
        gray = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return gray

def run_ocr(img: Image.Image) -> str:
    processed = preprocess(img)
    if OCR_ENGINE == "tesseract":
        import pytesseract
        # PSM 6: uniform block of text. PSM 11: sparse text (better for single fields)
        # Try both and return the longer result
        r6  = pytesseract.image_to_string(processed, config="--psm 6  --oem 3").strip()
        r11 = pytesseract.image_to_string(processed, config="--psm 11 --oem 3").strip()
        return r6 if len(r6) >= len(r11) else r11
    elif OCR_ENGINE == "easyocr":
        import numpy as np
        arr = np.array(processed.convert("RGB"))
        results = OCR_OBJ.readtext(arr, detail=0, paragraph=True)
        return " ".join(results).strip()
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
            if i + 1 < len(lines) and ':' not in lines[i+1] and len(lines[i+1]) < 80:
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

def pil_to_b64(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode()

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

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = dict(pages=[], pdf_texts=[], extracted={}, boxes=[],
                page_idx=0, active_field=None,
                fields=list(PRESETS["📄 Invoice"]),
                sel_x1=0, sel_y1=0, sel_x2=0, sel_y2=0)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clear_selection():
    st.session_state.sel_x1 = st.session_state.sel_y1 = 0
    st.session_state.sel_x2 = st.session_state.sel_y2 = 0

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 14px;border-bottom:1px solid #26262d;margin-bottom:20px">
  <div class="app-title">Doc<span>Extract</span></div>
  <div class="app-sub">Upload · Select a region · Extract with OCR — no API key needed</div>
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
                    pdf_text = (st.session_state.pdf_texts[idx]
                                if st.session_state.pdf_texts else "")
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
        val    = st.session_state.extracted.get(field, "")
        border = "#F59E0B" if field == st.session_state.active_field else "#26262d"
        arrow  = "→ " if field == st.session_state.active_field else ""
        st.markdown(f"""<div class="field-row" style="border-color:{border}">
          <div class="field-name">{arrow}{field}</div>
          <div class="field-val {'field-empty' if not val else ''}">{val or 'not extracted yet'}</div>
        </div>""", unsafe_allow_html=True)
        new_val = st.text_input("_", value=val, label_visibility="collapsed", key=f"inp_{field}")
        if new_val != val:
            st.session_state.extracted[field] = new_val
            st.rerun()

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

    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.extracted = {}
        st.session_state.boxes     = []
        clear_selection()
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
        clear_selection()

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
                clear_selection(); st.rerun()
        with c2:
            st.markdown(f'<div style="text-align:center;color:#6b6b78;font-size:12px;'
                        f'font-family:monospace;padding:8px">'
                        f'Page {st.session_state.page_idx+1} / {n}</div>',
                        unsafe_allow_html=True)
        with c3:
            if st.button("Next ▶") and st.session_state.page_idx < n - 1:
                st.session_state.page_idx += 1
                st.session_state.boxes = []
                clear_selection(); st.rerun()

    page_img     = st.session_state.pages[st.session_state.page_idx]
    display_img  = annotate(page_img, st.session_state.boxes)
    iw, ih       = page_img.size
    disp_w       = 680
    disp_h       = int(ih * disp_w / iw)
    b64          = pil_to_b64(display_img)
    canvas_key   = f"cv_{st.session_state.page_idx}"

    if st.session_state.active_field:
        st.markdown(f'<div class="infobox">🖱 Draw a box around '
                    f'<strong>{st.session_state.active_field}</strong></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="infobox">👈 Select a Target Field, '
                    'then draw a rectangle on the document.</div>', unsafe_allow_html=True)

    # ── Canvas ────────────────────────────────────────────────────────────────
    # The canvas writes x1/y1/x2/y2 into 4 hidden number inputs via DOM.
    # We give those inputs unique IDs so the JS can find them reliably.
    components.html(f"""<!DOCTYPE html><html><head><style>
    body{{margin:0;padding:0;background:#0c0c0f}}
    canvas{{border:1px solid #26262d;border-radius:10px;cursor:crosshair;display:block}}
    #msg{{font-family:monospace;font-size:11px;color:#6b6b78;margin-top:6px;
          padding:5px 8px;background:#141417;border-radius:5px;border:1px solid #26262d}}
    </style></head><body>
    <canvas id="c" width="{disp_w}" height="{disp_h}"></canvas>
    <div id="msg">🖱 Click and drag to select a region</div>
    <script>
    (function(){{
      const canvas=document.getElementById('c');
      const ctx=canvas.getContext('2d');
      const SX={iw}/{disp_w}, SY={ih}/{disp_h};
      let sx=0,sy=0,ex=0,ey=0,down=false;
      const img=new Image();
      img.onload=()=>ctx.drawImage(img,0,0,{disp_w},{disp_h});
      img.src='data:image/jpeg;base64,{b64}';

      function draw(){{
        ctx.clearRect(0,0,canvas.width,canvas.height);
        ctx.drawImage(img,0,0,{disp_w},{disp_h});
        const rx=Math.min(sx,ex),ry=Math.min(sy,ey),rw=Math.abs(ex-sx),rh=Math.abs(ey-sy);
        ctx.fillStyle='rgba(245,158,11,0.15)';ctx.fillRect(rx,ry,rw,rh);
        ctx.strokeStyle='#F59E0B';ctx.lineWidth=2;ctx.strokeRect(rx,ry,rw,rh);
      }}

      function xy(e){{
        const r=canvas.getBoundingClientRect(),s=e.touches?e.touches[0]:e;
        return [s.clientX-r.left, s.clientY-r.top];
      }}

      // Write value into a Streamlit number input identified by its data-testid label
      function setNum(idx, val){{
        // Streamlit renders number inputs as <input type="number"> in the parent frame
        const inputs = window.parent.document.querySelectorAll('input[type="number"]');
        if(inputs.length <= idx) return;
        const inp = inputs[idx];
        const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
        setter.call(inp, String(val));
        inp.dispatchEvent(new Event('input',{{bubbles:true}}));
        inp.dispatchEvent(new Event('change',{{bubbles:true}}));
      }}

      function done(){{
        down=false;
        const x1=Math.round(Math.min(sx,ex)*SX), y1=Math.round(Math.min(sy,ey)*SY);
        const x2=Math.round(Math.max(sx,ex)*SX), y2=Math.round(Math.max(sy,ey)*SY);
        if(Math.abs(x2-x1)<8||Math.abs(y2-y1)<8)return;
        document.getElementById('msg').textContent=
          '✓ x1='+x1+' y1='+y1+' x2='+x2+' y2='+y2+' — scroll down and click Extract';
        // coords map to the 4 number inputs rendered below the canvas in order
        setNum(0,x1); setNum(1,y1); setNum(2,x2); setNum(3,y2);
      }}

      canvas.addEventListener('mousedown', e=>{{[sx,sy]=xy(e);down=true;e.preventDefault();}});
      canvas.addEventListener('mousemove', e=>{{if(!down)return;[ex,ey]=xy(e);draw();}});
      canvas.addEventListener('mouseup',   e=>{{if(!down)return;[ex,ey]=xy(e);draw();done();}});
      canvas.addEventListener('touchstart',e=>{{[sx,sy]=xy(e);down=true;e.preventDefault();}},{{passive:false}});
      canvas.addEventListener('touchmove', e=>{{if(!down)return;[ex,ey]=xy(e);draw();e.preventDefault();}},{{passive:false}});
      canvas.addEventListener('touchend',  e=>{{if(!down)return;[ex,ey]=xy(e);draw();done();}});
    }})();
    </script></body></html>""", height=disp_h + 50, scrolling=False)

    # ── Coordinate inputs (persisted in session state) ────────────────────────
    st.markdown('<div class="panel-label" style="margin-top:10px">📐 Region Coordinates '
                '<span style="color:#3a3a3e;font-size:9px">(auto-filled by drawing above)</span></div>',
                unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1: x1 = st.number_input("x1", 0, iw, st.session_state.sel_x1, 1, key="ni_x1")
    with cc2: y1 = st.number_input("y1", 0, ih, st.session_state.sel_y1, 1, key="ni_y1")
    with cc3: x2 = st.number_input("x2", 0, iw, st.session_state.sel_x2, 1, key="ni_x2")
    with cc4: y2 = st.number_input("y2", 0, ih, st.session_state.sel_y2, 1, key="ni_y2")

    # Persist coords so the Extract button stays visible across reruns
    st.session_state.sel_x1 = x1
    st.session_state.sel_y1 = y1
    st.session_state.sel_x2 = x2
    st.session_state.sel_y2 = y2

    has_sel = (x2 > x1 + 8) and (y2 > y1 + 8)

    if has_sel:
        pad  = 6
        crop = page_img.crop((max(0, x1-pad), max(0, y1-pad),
                               min(iw, x2+pad), min(ih, y2+pad)))

        st.markdown('<div class="panel-label" style="margin-top:10px">🔎 Region Preview</div>',
                    unsafe_allow_html=True)
        # Show crop at 2x for visibility
        preview_w = min(crop.width * 2, 600)
        st.image(crop, width=preview_w)

        # ── EXTRACT BUTTON ────────────────────────────────────────────────────
        btn_label = (f"🔍 Extract into  \"{st.session_state.active_field}\""
                     if st.session_state.active_field else "🔍 Extract Text from Region")

        if st.button(btn_label, use_container_width=True, type="primary"):
            with st.spinner("Running OCR…"):
                try:
                    text = run_ocr(crop)
                    if not text:
                        st.warning("No text detected — try selecting a larger region, "
                                   "or use ⚡ Auto-Extract All.")
                    else:
                        st.success(f"✅  `{text[:120]}{'…' if len(text)>120 else ''}`")
                        if st.session_state.active_field:
                            st.session_state.extracted[st.session_state.active_field] = text
                            st.session_state.boxes.append(
                                {"x1":x1,"y1":y1,"x2":x2,"y2":y2})
                            # Advance to next field automatically
                            fields = st.session_state.fields
                            if st.session_state.active_field in fields:
                                idx = fields.index(st.session_state.active_field)
                                if idx + 1 < len(fields):
                                    st.session_state.active_field = fields[idx + 1]
                            clear_selection()
                            st.rerun()
                except Exception as e:
                    st.error(f"OCR error: {e}")
    else:
        st.caption("Draw a rectangle on the document above — coordinates will fill automatically.")
