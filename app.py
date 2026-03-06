import streamlit as st
import json
import io
import os
import re
import tempfile
import fitz
from PIL import Image, ImageDraw, ImageFilter
import streamlit.components.v1 as components
import base64

# ── Page config ───────────────────────────────────────────────────────────────
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
@st.cache_resource
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

def pil_to_b64(img, fmt="JPEG", quality=88):
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode()

def preprocess(img):
    gray = img.convert("L")
    sharp = gray.filter(ImageFilter.SHARPEN)
    w, h = sharp.size
    if w < 200 or h < 30:
        scale = max(200/w, 30/h, 2.5)
        sharp = sharp.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return sharp

def run_ocr(img):
    processed = preprocess(img)
    if OCR_ENGINE == "tesseract":
        import pytesseract
        return pytesseract.image_to_string(processed, config="--psm 6 --oem 3").strip()
    elif OCR_ENGINE == "easyocr":
        import numpy as np
        arr = np.array(processed.convert("RGB"))
        return " ".join(OCR_OBJ.readtext(arr, detail=0, paragraph=True)).strip()
    return ""

def pdf_to_pages(path):
    doc = fitz.open(path)
    pages, texts = [], []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        texts.append(page.get_text())
    return pages, texts

def smart_match(full_text, field):
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    fkey = field.lower().replace(" ", "")
    for i, line in enumerate(lines):
        if fkey in line.lower().replace(" ", ""):
            parts = re.split(r':\s*', line, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
            if i + 1 < len(lines) and ":" not in lines[i+1] and len(lines[i+1]) < 80:
                return lines[i+1]
    words = [w for w in field.lower().split() if len(w) > 3]
    for line in lines:
        if all(w in line.lower() for w in words):
            parts = re.split(r':\s*', line, maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
    return ""

def auto_extract(img, fields, pdf_text=""):
    text = pdf_text.strip() if pdf_text.strip() else run_ocr(img)
    if not fields:
        result = {}
        for line in text.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                k, v = k.strip(), v.strip()
                if k and v and len(k) < 50:
                    result[k] = v
        return result
    return {f: smart_match(text, f) for f in fields}

def annotate(img, boxes):
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for b in boxes:
        draw.rectangle([b["x1"],b["y1"],b["x2"],b["y2"]],
                       outline="#F59E0B", width=3, fill=(245,158,11,35))
    return out

PRESETS = {
    "📄 Invoice": ["Vendor Name","Invoice Number","Invoice Date","Due Date",
                   "Subtotal","Tax Amount","Total Amount","Payment Terms","PO Number"],
    "🪪 ID / Passport": ["Full Name","Date of Birth","Gender","Document Number",
                          "Nationality","Issue Date","Expiry Date","Place of Birth"],
    "🧾 Receipt": ["Store Name","Store Address","Date","Time",
                   "Items","Subtotal","Tax","Total","Payment Method"],
    "📝 Contract": ["Party A","Party B","Contract Date","Effective Date",
                    "Contract Value","Duration","Jurisdiction"],
    "🔑 Generic Key-Value": [],
    "✏️ Custom": [],
}

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("pages",[]),("pdf_texts",[]),("extracted",{}),("boxes",[]),
              ("page_idx",0),("active_field",None),("fields",list(PRESETS["📄 Invoice"])),
              ("pending_crop",None),("sel_x1",0),("sel_y1",0),("sel_x2",0),("sel_y2",0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 14px;border-bottom:1px solid #26262d;margin-bottom:20px">
  <div class="app-title">Doc<span>Extract</span></div>
  <div class="app-sub">Upload · Draw a region · Extract with OCR — no API key needed</div>
</div>
""", unsafe_allow_html=True)

engine_color = {"tesseract":"#34d399","easyocr":"#818cf8","pymupdf":"#F59E0B"}.get(OCR_ENGINE,"#aaa")
st.markdown(f'<span style="font-family:monospace;font-size:11px;color:{engine_color}">● {OCR_ENGINE.upper()} active</span>',
            unsafe_allow_html=True)

left, right = st.columns([1, 1.9], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT
# ══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="panel-label">📋 Template</div>', unsafe_allow_html=True)
    preset = st.selectbox("Template", list(PRESETS.keys()), label_visibility="collapsed")
    if st.button("Load Template"):
        st.session_state.fields = list(PRESETS[preset])
        st.session_state.extracted = {}
        st.rerun()

    with st.expander("✏️ Edit Fields"):
        fields_text = st.text_area("One per line", value="\n".join(st.session_state.fields),
                                    height=140, label_visibility="collapsed")
        if st.button("Update Fields"):
            st.session_state.fields = [f.strip() for f in fields_text.splitlines() if f.strip()]
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">🎯 Target Field</div>', unsafe_allow_html=True)

    field_opts = ["— select —"] + st.session_state.fields
    cur_idx = (st.session_state.fields.index(st.session_state.active_field) + 1
               if st.session_state.active_field in st.session_state.fields else 0)
    active = st.selectbox("Field", field_opts, index=cur_idx, label_visibility="collapsed")
    st.session_state.active_field = None if active == "— select —" else active

    if st.session_state.pages:
        if st.button("⚡ Auto-Extract All", use_container_width=True):
            with st.spinner("Running OCR…"):
                try:
                    idx = st.session_state.page_idx
                    pdf_text = st.session_state.pdf_texts[idx] if st.session_state.pdf_texts else ""
                    result = auto_extract(st.session_state.pages[idx],
                                         st.session_state.fields, pdf_text)
                    st.session_state.extracted.update(result)
                    st.success(f"✅ Extracted {len(result)} fields")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">📊 Extracted Data</div>', unsafe_allow_html=True)

    for field in st.session_state.fields:
        val = st.session_state.extracted.get(field, "")
        border = "#F59E0B" if field == st.session_state.active_field else "#26262d"
        arrow = "→ " if field == st.session_state.active_field else ""
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
        st.download_button("⬇ JSON", json.dumps(st.session_state.extracted, indent=2),
                            "extracted.json", "application/json", use_container_width=True)
    with c2:
        csv_str = "\n".join(["Field,Value"] + [f'"{k}","{v}"'
                             for k,v in st.session_state.extracted.items()])
        st.download_button("⬇ CSV", csv_str, "extracted.csv", "text/csv", use_container_width=True)

    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.extracted = {}
        st.session_state.boxes = []
        st.session_state.pending_crop = None
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
        ext = uploaded.name.rsplit(".",1)[-1].lower()
        if ext == "pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data); tmp_path = tmp.name
            pages, texts = pdf_to_pages(tmp_path)
            os.unlink(tmp_path)
            st.session_state.pages = pages
            st.session_state.pdf_texts = texts
        else:
            st.session_state.pages = [Image.open(io.BytesIO(data)).convert("RGB")]
            st.session_state.pdf_texts = []
        st.session_state.page_idx = 0
        st.session_state.boxes = []
        st.session_state.pending_crop = None
        st.session_state.sel_x1 = st.session_state.sel_y1 = 0
        st.session_state.sel_x2 = st.session_state.sel_y2 = 0

    if not st.session_state.pages:
        st.markdown("""<div style="border:2px dashed #26262d;border-radius:12px;padding:60px;text-align:center">
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
                st.session_state.pending_crop = None
                st.session_state.sel_x1 = st.session_state.sel_y1 = 0
                st.session_state.sel_x2 = st.session_state.sel_y2 = 0
                st.rerun()
        with c2:
            st.markdown(f'<div style="text-align:center;color:#6b6b78;font-size:12px;font-family:monospace;padding:8px">'
                        f'Page {st.session_state.page_idx+1} / {n}</div>', unsafe_allow_html=True)
        with c3:
            if st.button("Next ▶") and st.session_state.page_idx < n-1:
                st.session_state.page_idx += 1
                st.session_state.boxes = []
                st.session_state.pending_crop = None
                st.session_state.sel_x1 = st.session_state.sel_y1 = 0
                st.session_state.sel_x2 = st.session_state.sel_y2 = 0
                st.rerun()

    page_img = st.session_state.pages[st.session_state.page_idx]

    if st.session_state.active_field:
        st.markdown(f'<div class="infobox">🖱 Draw a box around <strong>{st.session_state.active_field}</strong>'
                    f' → the crop will appear below → click <strong>Extract Text</strong></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="infobox">👈 Select a Target Field, then draw a rectangle on the document.</div>',
                    unsafe_allow_html=True)

    # ── Canvas: show annotated image, capture coords via number inputs ────────
    display_img = annotate(page_img, st.session_state.boxes)
    iw, ih = display_img.size
    disp_w = 680
    disp_h = int(ih * disp_w / iw)
    b64 = pil_to_b64(display_img)

    # Unique key per page so canvas resets on page change
    canvas_key = f"cv_{st.session_state.page_idx}"

    canvas_html = f"""
<div style="background:#0c0c0f">
<canvas id="{canvas_key}" width="{disp_w}" height="{disp_h}"
  style="border:1px solid #26262d;border-radius:10px;cursor:crosshair;display:block"></canvas>
<div id="info_{canvas_key}" style="font-family:monospace;font-size:11px;color:#6b6b78;
  margin-top:6px;padding:5px 8px;background:#141417;border-radius:5px">
  🖱 Click and drag to draw a selection rectangle
</div>
</div>
<script>
(function() {{
  const CK = "{canvas_key}";
  const canvas = document.getElementById(CK);
  const ctx = canvas.getContext('2d');
  const scaleX = {iw} / {disp_w};
  const scaleY = {ih} / {disp_h};
  let sx=0,sy=0,ex=0,ey=0,down=false;
  const imgEl = new Image();
  imgEl.onload = () => ctx.drawImage(imgEl,0,0,{disp_w},{disp_h});
  imgEl.src = 'data:image/jpeg;base64,{b64}';

  function redraw(rx,ry,rw,rh) {{
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(imgEl,0,0,{disp_w},{disp_h});
    ctx.fillStyle='rgba(245,158,11,0.15)';
    ctx.fillRect(rx,ry,rw,rh);
    ctx.strokeStyle='#F59E0B';ctx.lineWidth=2;
    ctx.strokeRect(rx,ry,rw,rh);
  }}

  function getXY(e) {{
    const r=canvas.getBoundingClientRect();
    const s=e.touches?e.touches[0]:e;
    return [s.clientX-r.left, s.clientY-r.top];
  }}

  canvas.addEventListener('mousedown', e=>{{
    [sx,sy]=getXY(e); down=true; e.preventDefault();
  }});
  canvas.addEventListener('mousemove', e=>{{
    if(!down) return;
    [ex,ey]=getXY(e);
    redraw(Math.min(sx,ex),Math.min(sy,ey),Math.abs(ex-sx),Math.abs(ey-sy));
  }});
  canvas.addEventListener('mouseup', e=>{{
    if(!down) return; down=false;
    [ex,ey]=getXY(e);
    const x1=Math.round(Math.min(sx,ex)*scaleX), y1=Math.round(Math.min(sy,ey)*scaleY);
    const x2=Math.round(Math.max(sx,ex)*scaleX), y2=Math.round(Math.max(sy,ey)*scaleY);
    if(Math.abs(x2-x1)>8 && Math.abs(y2-y1)>8) {{
      document.getElementById('info_'+CK).textContent =
        '✓ Selected: x1='+x1+' y1='+y1+' x2='+x2+' y2='+y2+' — now enter coords below and click Extract';
      // Fill the number inputs Streamlit rendered
      function setInput(label, val) {{
        const inputs = window.parent.document.querySelectorAll('input[type=number]');
        const map = {{'x1':0,'y1':1,'x2':2,'y2':3}};
        const i = map[label];
        if(inputs[i]===undefined) return;
        const nv = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;
        nv.call(inputs[i], val);
        inputs[i].dispatchEvent(new Event('input',{{bubbles:true}}));
      }}
      setInput('x1',x1); setInput('y1',y1); setInput('x2',x2); setInput('y2',y2);
    }}
  }});
  canvas.addEventListener('touchstart',e=>{{[sx,sy]=getXY(e);down=true;e.preventDefault();}},{{passive:false}});
  canvas.addEventListener('touchmove', e=>{{if(!down)return;[ex,ey]=getXY(e);redraw(Math.min(sx,ex),Math.min(sy,ey),Math.abs(ex-sx),Math.abs(ey-sy));e.preventDefault();}},{{passive:false}});
  canvas.addEventListener('touchend',  e=>{{if(!down)return;down=false;[ex,ey]=getXY(e);
    const x1=Math.round(Math.min(sx,ex)*scaleX),y1=Math.round(Math.min(sy,ey)*scaleY);
    const x2=Math.round(Math.max(sx,ex)*scaleX),y2=Math.round(Math.max(sy,ey)*scaleY);
    if(Math.abs(x2-x1)>8&&Math.abs(y2-y1)>8){{
      function setInput(label,val){{const inputs=window.parent.document.querySelectorAll('input[type=number]');const map={{'x1':0,'y1':1,'x2':2,'y2':3}};const i=map[label];if(inputs[i]===undefined)return;const nv=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,'value').set;nv.call(inputs[i],val);inputs[i].dispatchEvent(new Event('input',{{bubbles:true}}));}}
      setInput('x1',x1);setInput('y1',y1);setInput('x2',x2);setInput('y2',y2);
    }}
  }});
}})();
</script>
"""
    components.html(canvas_html, height=disp_h + 50, scrolling=False)

    # ── Coordinate inputs (visible so user can see/adjust) ────────────────────
    st.markdown('<div class="panel-label" style="margin-top:12px">📐 Selection Coordinates</div>',
                unsafe_allow_html=True)
    cc1, cc2, cc3, cc4 = st.columns(4)
    with cc1: x1 = st.number_input("x1", min_value=0, max_value=iw, value=st.session_state.sel_x1, step=1, key="ni_x1")
    with cc2: y1 = st.number_input("y1", min_value=0, max_value=ih, value=st.session_state.sel_y1, step=1, key="ni_y1")
    with cc3: x2 = st.number_input("x2", min_value=0, max_value=iw, value=st.session_state.sel_x2, step=1, key="ni_x2")
    with cc4: y2 = st.number_input("y2", min_value=0, max_value=ih, value=st.session_state.sel_y2, step=1, key="ni_y2")
    # Persist coords in session state so button survives reruns
    st.session_state.sel_x1 = x1
    st.session_state.sel_y1 = y1
    st.session_state.sel_x2 = x2
    st.session_state.sel_y2 = y2

    # Show crop preview when valid selection exists
    has_selection = (x2 > x1 + 8) and (y2 > y1 + 8)

    if has_selection:
        pad = 4
        crop = page_img.crop((max(0,x1-pad), max(0,y1-pad),
                               min(iw,x2+pad), min(ih,y2+pad)))
        st.markdown('<div class="panel-label" style="margin-top:8px">🔎 Region Preview</div>',
                    unsafe_allow_html=True)
        st.image(crop, use_container_width=False, width=min(crop.width, 500))

        if st.session_state.active_field:
            btn_label = f"🔍 Extract  \"{st.session_state.active_field}\""
        else:
            btn_label = "🔍 Extract Text from Region"

        if st.button(btn_label, use_container_width=True, type="primary"):
            with st.spinner("Running OCR on selected region…"):
                try:
                    text = run_ocr(crop)
                    if not text:
                        st.warning("⚠ No text detected. Try selecting a larger area, or use Auto-Extract All.")
                    else:
                        st.success(f"✅ Extracted: **{text[:80]}{'…' if len(text)>80 else ''}**")
                        if st.session_state.active_field:
                            st.session_state.extracted[st.session_state.active_field] = text
                            st.session_state.boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
                            # Advance to next field
                            fields = st.session_state.fields
                            if st.session_state.active_field in fields:
                                idx = fields.index(st.session_state.active_field)
                                if idx + 1 < len(fields):
                                    st.session_state.active_field = fields[idx+1]
                            st.session_state.sel_x1 = st.session_state.sel_y1 = 0
                            st.session_state.sel_x2 = st.session_state.sel_y2 = 0
                            st.rerun()
                except Exception as e:
                    st.error(f"OCR error: {e}")
    else:
        st.caption("Draw a rectangle on the document above to select a region.")
