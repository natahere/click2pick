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
.panel-label{font-family:'JetBrains Mono',monospace;font-size:10px;color:#6b6b78;
    letter-spacing:2px;text-transform:uppercase;margin-bottom:8px}
.infobox{background:#13130f;border-left:3px solid #F59E0B;border-radius:6px;
    padding:9px 13px;font-size:12px;color:#a89060;margin-bottom:10px}
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
def preprocess(img):
    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.SHARPEN)
    w, h = gray.size
    if w < 400:
        s = 400 / w
        gray = gray.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return gray

def run_ocr(img):
    p   = preprocess(img)
    r6  = pytesseract.image_to_string(p, config="--psm 6  --oem 3").strip()
    r11 = pytesseract.image_to_string(p, config="--psm 11 --oem 3").strip()
    return r6 if len(r6) >= len(r11) else r11

def pdf_to_pages(path):
    doc = fitz.open(path)
    pages, texts = [], []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        texts.append(page.get_text())
    return pages, texts

def smart_match(text, field):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
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

def auto_extract(img, fields, pdf_text=""):
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

def pil_to_b64(img, quality=88):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode()

def annotate(img, boxes):
    out  = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for b in boxes:
        draw.rectangle([b["x1"],b["y1"],b["x2"],b["y2"]],
                       outline="#34d399", width=2, fill=(52,211,153,30))
    return out

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

DISP_W = 680

DEFAULTS = dict(pages=[], pdf_texts=[], extracted={}, boxes=[],
                page_idx=0, active_field=None,
                fields=list(PRESETS["📄 Invoice"]),
                sel=None)            # sel = {x1,y1,x2,y2} in original image coords
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Read selection posted via query params ────────────────────────────────────
# The canvas iframe cannot postMessage to parent (CSP blocked).
# Instead it submits a tiny HTML form whose action is the current page URL
# with ?sel=x1,y1,x2,y2 appended. The browser navigates the TOP frame,
# Streamlit picks up the query param, stores it, clears it, reruns cleanly.
params = st.query_params
if "sel" in params:
    try:
        parts = [int(v) for v in params["sel"].split(",")]
        if len(parts) == 4:
            st.session_state.sel = {"x1":parts[0],"y1":parts[1],
                                     "x2":parts[2],"y2":parts[3]}
    except Exception:
        pass
    st.query_params.clear()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 14px;border-bottom:1px solid #26262d;margin-bottom:20px">
  <div class="app-title">Doc<span>Extract</span></div>
  <div style="font-size:12px;color:#6b6b78;margin-top:3px">
    Upload · Drag on the image to select · Click Extract</div>
</div>""", unsafe_allow_html=True)

left, right = st.columns([1, 1.9], gap="large")

# ══════════════ LEFT ══════════════════════════════════════════════════════════
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
            with st.spinner("Running OCR…"):
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
        val    = st.session_state.extracted.get(field, "")
        is_act = field == st.session_state.active_field
        label  = f"**→ {field}**" if is_act else field
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
        st.session_state.sel       = None
        st.rerun()

# ══════════════ RIGHT ═════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="panel-label">📄 Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload",
                                  type=["pdf","png","jpg","jpeg","webp","tiff","bmp"],
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
        st.session_state.sel      = None

    if not st.session_state.pages:
        st.markdown("""<div style="border:2px dashed #26262d;border-radius:12px;
            padding:60px;text-align:center">
          <div style="font-size:48px">📄</div>
          <div style="color:#6b6b78;margin-top:12px">Upload a PDF or image to get started</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    n = len(st.session_state.pages)
    if n > 1:
        c1, c2, c3 = st.columns([1,2,1])
        with c1:
            if st.button("◀ Prev") and st.session_state.page_idx > 0:
                st.session_state.page_idx -= 1
                st.session_state.boxes = []
                st.session_state.sel   = None
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
                st.session_state.sel   = None
                st.rerun()

    page_img = st.session_state.pages[st.session_state.page_idx]
    iw, ih   = page_img.size
    disp_h   = int(ih * DISP_W / iw)
    scaleX   = iw / DISP_W
    scaleY   = ih / disp_h

    # Instruction
    sel = st.session_state.sel
    if not st.session_state.active_field:
        st.markdown('<div class="infobox">👈 Select a <strong>Target Field</strong> on the left first.</div>',
                    unsafe_allow_html=True)
    elif sel is None:
        st.markdown(f'<div class="infobox">🖱 <strong>Click and drag</strong> on the document '
                    f'to select the region for '
                    f'<strong>{st.session_state.active_field}</strong></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="infobox">✅ Region selected — '
                    f'click <strong>Extract</strong> below, or drag again to re-select.</div>',
                    unsafe_allow_html=True)

    # ── Canvas rendered via components.html ───────────────────────────────────
    # Boxes in display coords for drawing
    disp_boxes = [{"x1":int(b["x1"]/scaleX),"y1":int(b["y1"]/scaleY),
                   "x2":int(b["x2"]/scaleX),"y2":int(b["y2"]/scaleY)}
                  for b in st.session_state.boxes]

    # Render current selection on the image itself (no JS needed for this part)
    display_img = annotate(page_img, st.session_state.boxes)
    if sel:
        draw = ImageDraw.Draw(display_img, "RGBA")
        draw.rectangle([sel["x1"],sel["y1"],sel["x2"],sel["y2"]],
                       outline="#F59E0B", width=3, fill=(245,158,11,40))
    b64 = pil_to_b64(display_img.resize((DISP_W, disp_h), Image.LANCZOS))

    # The current page URL (without query params) — used as the form action
    # We encode selection as ?sel=x1,y1,x2,y2 which Streamlit reads above
    canvas_html = f"""
<!DOCTYPE html><html><head>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#0c0c0f;overflow:hidden;user-select:none}}
  canvas{{display:block;cursor:crosshair}}
  #msg{{font-family:monospace;font-size:11px;color:#6b6b78;
        padding:5px 8px;background:#141417;border-top:1px solid #26262d}}
  form{{display:none}}
</style></head><body>
<canvas id="c" width="{DISP_W}" height="{disp_h}"></canvas>
<div id="msg">🖱 Click and drag to select a region, then click Extract below</div>
<form id="f" method="GET">
  <input type="hidden" id="sel_input" name="sel" value="">
</form>
<script>
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const msg    = document.getElementById('msg');
const form   = document.getElementById('f');
const selInp = document.getElementById('sel_input');

const SX = {scaleX};  // display → original
const SY = {scaleY};

let sx=0, sy=0, dragging=false;

// Load image
const img = new Image();
img.onload = () => ctx.drawImage(img, 0, 0, {DISP_W}, {disp_h});
img.src = 'data:image/jpeg;base64,{b64}';

function redraw(rx, ry, rw, rh) {{
  ctx.clearRect(0,0,{DISP_W},{disp_h});
  ctx.drawImage(img, 0, 0, {DISP_W}, {disp_h});
  if (rw>0 && rh>0) {{
    ctx.fillStyle = 'rgba(245,158,11,0.2)';
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeStyle = '#F59E0B';
    ctx.lineWidth   = 2;
    ctx.setLineDash([6,3]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
  }}
}}

function getPos(e) {{
  const r = canvas.getBoundingClientRect();
  const s = e.touches ? e.touches[0] : e;
  return [s.clientX - r.left, s.clientY - r.top];
}}

function finish(ex, ey) {{
  dragging = false;
  const x1 = Math.round(Math.min(sx,ex) * SX);
  const y1 = Math.round(Math.min(sy,ey) * SY);
  const x2 = Math.round(Math.max(sx,ex) * SX);
  const y2 = Math.round(Math.max(sy,ey) * SY);
  redraw(Math.min(sx,ex), Math.min(sy,ey), Math.abs(ex-sx), Math.abs(ey-sy));
  if (Math.abs(x2-x1) > 5 && Math.abs(y2-y1) > 5) {{
    msg.textContent = '✓ Region selected — click Extract below';
    // Submit form to parent page URL with ?sel= param
    // This navigates the TOP frame, which Streamlit intercepts
    selInp.value = x1+','+y1+','+x2+','+y2;
    form.action  = window.parent.location.href.split('?')[0];
    form.target  = '_parent';
    form.submit();
  }} else {{
    msg.textContent = '⚠ Selection too small — try again';
  }}
}}

canvas.addEventListener('mousedown', e => {{
  [sx,sy] = getPos(e); dragging=true; e.preventDefault();
}});
canvas.addEventListener('mousemove', e => {{
  if (!dragging) return;
  const [ex,ey] = getPos(e);
  redraw(Math.min(sx,ex),Math.min(sy,ey),Math.abs(ex-sx),Math.abs(ey-sy));
}});
canvas.addEventListener('mouseup', e => {{
  if (!dragging) return;
  const [ex,ey] = getPos(e); finish(ex,ey);
}});
canvas.addEventListener('touchstart', e=>{{
  [sx,sy]=getPos(e); dragging=true; e.preventDefault();
}},{{passive:false}});
canvas.addEventListener('touchmove', e=>{{
  if(!dragging)return;
  const [ex,ey]=getPos(e);
  redraw(Math.min(sx,ex),Math.min(sy,ey),Math.abs(ex-sx),Math.abs(ey-sy));
  e.preventDefault();
}},{{passive:false}});
canvas.addEventListener('touchend', e=>{{
  if(!dragging)return;
  const [ex,ey]=getPos({{touches:e.changedTouches}}); finish(ex,ey);
}});
</script>
</body></html>"""

    import streamlit.components.v1 as components
    components.html(canvas_html, height=disp_h + 30, scrolling=False)

    # ── Crop preview + Extract ─────────────────────────────────────────────────
    if sel and sel["x2"]-sel["x1"] > 8 and sel["y2"]-sel["y1"] > 8:
        pad  = 6
        crop = page_img.crop((max(0, sel["x1"]-pad), max(0, sel["y1"]-pad),
                               min(iw, sel["x2"]+pad), min(ih, sel["y2"]+pad)))

        st.markdown('<div class="panel-label" style="margin-top:10px">🔎 Crop Preview</div>',
                    unsafe_allow_html=True)
        st.image(crop, width=min(crop.width * 2, DISP_W))

        btn_col, reset_col = st.columns([3, 1])
        with btn_col:
            lbl = (f"🔍 Extract → \"{st.session_state.active_field}\""
                   if st.session_state.active_field else "🔍 Extract Text")
            if st.button(lbl, use_container_width=True, type="primary"):
                with st.spinner("Running OCR…"):
                    try:
                        text = run_ocr(crop)
                        if not text:
                            st.warning("No text detected — try a wider selection.")
                        else:
                            if st.session_state.active_field:
                                field = st.session_state.active_field
                                st.session_state.extracted[field] = text
                                st.session_state.boxes.append({
                                    "x1":sel["x1"],"y1":sel["y1"],
                                    "x2":sel["x2"],"y2":sel["y2"],
                                })
                                fields = st.session_state.fields
                                if field in fields:
                                    idx = fields.index(field)
                                    if idx+1 < len(fields):
                                        st.session_state.active_field = fields[idx+1]
                                st.session_state.sel = None
                                st.rerun()
                            else:
                                st.success(f"✅ `{text}`")
                    except Exception as e:
                        st.error(f"OCR error: {e}")
        with reset_col:
            if st.button("↺ Reset", use_container_width=True):
                st.session_state.sel = None
                st.rerun()
