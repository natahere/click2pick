import streamlit as st
import anthropic
import base64
import json
import io
import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocExtract",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=JetBrains+Mono:wght@400;500&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] { font-family: 'Lato', sans-serif !important; }
.stApp { background: #0c0c0f; color: #e2dfd8; }

.main-header { padding: 24px 0 16px; border-bottom: 1px solid #26262d; margin-bottom: 24px; }
.app-title { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; color: #fff; letter-spacing: -0.5px; }
.app-title span { color: #F59E0B; }
.app-sub { font-size: 13px; color: #6b6b78; margin-top: 4px; font-weight: 300; }

.panel-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500;
    color: #6b6b78; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px; }

.instruction-box { background: #13130f; border: 1px solid #2a2600; border-left: 3px solid #F59E0B;
    border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #a89060; margin-bottom: 12px; }

.field-row { background: #141417; border: 1px solid #26262d; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px; }
.field-name { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #6b6b78;
    text-transform: uppercase; letter-spacing: 1px; }
.field-val { font-size: 14px; color: #e2dfd8; margin-top: 3px; min-height: 18px; }
.field-empty { color: #2a2a2e; font-style: italic; font-size: 12px; }

.stButton > button { background: #F59E0B !important; color: #0c0c0f !important;
    border: none !important; border-radius: 8px !important; font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important; }
.stButton > button:hover { background: #d97706 !important; }

.stSelectbox > div > div, .stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #1c1c21 !important; border: 1px solid #26262d !important;
    border-radius: 8px !important; color: #e2dfd8 !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }
hr { border-color: #1e1e22 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode()

def pdf_to_pages(path):
    doc = fitz.open(path)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return pages

def get_client(api_key):
    key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError("Please enter your Anthropic API key.")
    return anthropic.Anthropic(api_key=key)

def extract_region(img, x1, y1, x2, y2, field_name, api_key):
    crop = img.crop((x1, y1, x2, y2))
    client = get_client(api_key)
    resp = client.messages.create(
        model="claude-opus-4-5", max_tokens=256,
        messages=[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":pil_to_b64(crop)}},
            {"type":"text","text":f"Extract ONLY the text in this image for field '{field_name}'. Return raw text only, no explanation."}
        ]}]
    )
    return resp.content[0].text.strip()

def auto_extract(img, fields, api_key):
    client = get_client(api_key)
    prompt = (
        "Extract ALL key-value pairs from this document. Return ONLY flat JSON, no markdown."
        if not fields else
        f"Extract these fields: {json.dumps(fields)}. Return ONLY flat JSON with those exact keys. Empty string if not found. No markdown."
    )
    resp = client.messages.create(
        model="claude-opus-4-5", max_tokens=2048,
        messages=[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/png","data":pil_to_b64(img)}},
            {"type":"text","text":prompt}
        ]}]
    )
    raw = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
    return json.loads(raw)

def annotate(img, boxes):
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    for b in boxes:
        draw.rectangle([b["x1"],b["y1"],b["x2"],b["y2"]],
                       outline="#F59E0B", width=3, fill=(245,158,11,35))
    return out

# ── Custom HTML canvas selector ───────────────────────────────────────────────
def image_selector(img, key):
    """
    Renders document as an HTML5 canvas with drag-to-select.
    Communicates selection back via Streamlit query params.
    Returns dict {x1,y1,x2,y2} in image pixels, or None.
    """
    iw, ih = img.size
    disp_w = 680
    disp_h = int(ih * disp_w / iw)
    b64 = pil_to_b64(img)

    # Read any existing selection for this key
    params = st.query_params
    sel_key = f"sel_{key}"
    current_sel = None
    if sel_key in params:
        try:
            current_sel = json.loads(params[sel_key])
        except Exception:
            pass

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; background:#0c0c0f; }}
  canvas {{ border:1px solid #26262d; border-radius:10px; cursor:crosshair; display:block; }}
  #info {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:#6b6b78;
           margin-top:8px; padding:6px 10px; background:#141417;
           border-radius:6px; border:1px solid #26262d; }}
  #btn {{ display:none; margin-top:8px; width:100%; padding:10px;
          background:#F59E0B; color:#0c0c0f; border:none; border-radius:8px;
          font-weight:700; font-size:13px; cursor:pointer; font-family:sans-serif; }}
  #btn:hover {{ background:#d97706; }}
</style>
</head>
<body>
<canvas id="canvas" width="{disp_w}" height="{disp_h}"></canvas>
<div id="info">🖱 Click and drag to select a region on the document</div>
<button id="btn" onclick="sendSelection()">✓ Use This Selection</button>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scaleX = {iw} / {disp_w};
const scaleY = {ih} / {disp_h};
const img = new Image();
let sx=0, sy=0, ex=0, ey=0, dragging=false, hasSelection=false;

img.onload = function() {{
  ctx.drawImage(img, 0, 0, {disp_w}, {disp_h});
  // Restore previous selection if any
  const prev = {json.dumps(current_sel) if current_sel else 'null'};
  if (prev) {{
    sx = prev.x1/scaleX; sy = prev.y1/scaleY;
    ex = prev.x2/scaleX; ey = prev.y2/scaleY;
    hasSelection = true;
    redraw();
    document.getElementById('btn').style.display='block';
    document.getElementById('info').textContent =
      `✓ Region: x1:${{prev.x1}} y1:${{prev.y1}} x2:${{prev.x2}} y2:${{prev.y2}}`;
  }}
}};
img.src = 'data:image/png;base64,{b64}';

function redraw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0,{disp_w},{disp_h});
  if (hasSelection) {{
    const rx=Math.min(sx,ex), ry=Math.min(sy,ey);
    const rw=Math.abs(ex-sx), rh=Math.abs(ey-sy);
    ctx.fillStyle='rgba(245,158,11,0.15)';
    ctx.fillRect(rx,ry,rw,rh);
    ctx.strokeStyle='#F59E0B'; ctx.lineWidth=2;
    ctx.strokeRect(rx,ry,rw,rh);
  }}
}}

function getPos(e) {{
  const r = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return {{ x: src.clientX-r.left, y: src.clientY-r.top }};
}}

canvas.addEventListener('mousedown', e=>{{ const p=getPos(e); sx=p.x; sy=p.y; dragging=true; hasSelection=false; e.preventDefault(); }});
canvas.addEventListener('mousemove', e=>{{ if(!dragging) return; const p=getPos(e); ex=p.x; ey=p.y; hasSelection=true; redraw(); }});
canvas.addEventListener('mouseup',   e=>{{ if(!dragging) return; dragging=false; const p=getPos(e); ex=p.x; ey=p.y; finishDrag(); }});
canvas.addEventListener('touchstart',e=>{{ const p=getPos(e); sx=p.x; sy=p.y; dragging=true; hasSelection=false; e.preventDefault(); }},{{passive:false}});
canvas.addEventListener('touchmove', e=>{{ if(!dragging) return; const p=getPos(e); ex=p.x; ey=p.y; hasSelection=true; redraw(); e.preventDefault(); }},{{passive:false}});
canvas.addEventListener('touchend',  e=>{{ if(!dragging) return; dragging=false; const p=getPos(e.changedTouches?{{touches:e.changedTouches}}:e); ex=p.x; ey=p.y; finishDrag(); }});

function finishDrag() {{
  hasSelection = true; redraw();
  const x1=Math.round(Math.min(sx,ex)*scaleX), y1=Math.round(Math.min(sy,ey)*scaleY);
  const x2=Math.round(Math.max(sx,ex)*scaleX), y2=Math.round(Math.max(sy,ey)*scaleY);
  if (Math.abs(x2-x1)>5 && Math.abs(y2-y1)>5) {{
    document.getElementById('info').textContent = `✓ Region selected: x1:${{x1}} y1:${{y1}} x2:${{x2}} y2:${{y2}} — click button to extract`;
    document.getElementById('btn').style.display='block';
    document.getElementById('btn').dataset.sel = JSON.stringify({{x1,y1,x2,y2}});
  }}
}}

function sendSelection() {{
  const sel = document.getElementById('btn').dataset.sel;
  if (!sel) return;
  // Post message to Streamlit parent
  window.parent.postMessage({{type:'docextract_sel', key:'{sel_key}', value:sel}}, '*');
}}
</script>
</body>
</html>
"""
    components.html(html_code, height=disp_h + 80, scrolling=False)
    return current_sel


# ── Streamlit message receiver (inject JS in main page) ───────────────────────
def inject_message_receiver():
    """Listen for postMessage from iframe and update query params via Streamlit."""
    st.markdown("""
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'docextract_sel') {
            const url = new URL(window.location.href);
            url.searchParams.set(e.data.key, e.data.value);
            window.location.href = url.toString();
        }
    });
    </script>
    """, unsafe_allow_html=True)

inject_message_receiver()

# ── Session state ─────────────────────────────────────────────────────────────
defaults = dict(pages=[], extracted={}, boxes=[], page_idx=0, active_field=None,
                api_key="", fields=list(PRESETS["📄 Invoice"]))
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="app-title">Doc<span>Extract</span></div>
  <div class="app-sub">Upload · Draw a region · Extract structured data with Claude Vision</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.9], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Config & Fields
# ══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div class="panel-label">⚙ Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API Key", type="password",
                             value=st.session_state.api_key, placeholder="sk-ant-api03-…")
    st.session_state.api_key = api_key

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">📋 Template</div>', unsafe_allow_html=True)
    preset = st.selectbox("Template", list(PRESETS.keys()), label_visibility="collapsed")
    if st.button("Load Template"):
        st.session_state.fields = list(PRESETS[preset])
        st.session_state.extracted = {}
        st.rerun()

    with st.expander("✏️ Edit Fields"):
        fields_text = st.text_area("One per line", value="\n".join(st.session_state.fields),
                                    height=150, label_visibility="collapsed")
        if st.button("Update Fields"):
            st.session_state.fields = [f.strip() for f in fields_text.splitlines() if f.strip()]
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">🎯 Target Field</div>', unsafe_allow_html=True)
    active = st.selectbox("Field to fill", ["— select —"] + st.session_state.fields,
                           label_visibility="collapsed",
                           index=(st.session_state.fields.index(st.session_state.active_field)+1
                                  if st.session_state.active_field in st.session_state.fields else 0))
    st.session_state.active_field = None if active == "— select —" else active

    if st.session_state.pages:
        if st.button("⚡ Auto-Extract All Fields", use_container_width=True):
            if not api_key:
                st.error("Enter your API key first.")
            else:
                with st.spinner("Extracting with Claude Vision…"):
                    try:
                        result = auto_extract(
                            st.session_state.pages[st.session_state.page_idx],
                            st.session_state.fields, api_key)
                        st.session_state.extracted.update(result)
                        st.success(f"✅ Extracted {len(result)} fields!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">📊 Extracted Data</div>', unsafe_allow_html=True)

    for field in st.session_state.fields:
        val = st.session_state.extracted.get(field, "")
        is_active = field == st.session_state.active_field
        border = "#F59E0B" if is_active else "#26262d"
        st.markdown(f"""
        <div class="field-row" style="border-color:{border}">
          <div class="field-name">{'→ ' if is_active else ''}{field}</div>
          <div class="field-val {'field-empty' if not val else ''}">{val if val else 'not extracted yet'}</div>
        </div>""", unsafe_allow_html=True)
        new_val = st.text_input("_", value=val, label_visibility="collapsed", key=f"inp_{field}")
        if new_val != val:
            st.session_state.extracted[field] = new_val
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label">💾 Export</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ JSON", json.dumps(st.session_state.extracted, indent=2),
                            "extracted.json", "application/json", use_container_width=True)
    with c2:
        csv_str = "\n".join(["Field,Value"]+[f'"{k}","{v}"'
                             for k,v in st.session_state.extracted.items()])
        st.download_button("⬇ CSV", csv_str, "extracted.csv", "text/csv", use_container_width=True)

    if st.button("🗑 Clear All", use_container_width=True):
        st.session_state.extracted = {}
        st.session_state.boxes = []
        st.query_params.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Document Viewer
# ══════════════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div class="panel-label">📄 Document</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload", type=["pdf","png","jpg","jpeg","webp","tiff","bmp"],
                                  label_visibility="collapsed")
    if uploaded:
        data = uploaded.read()
        ext = uploaded.name.rsplit(".",1)[-1].lower()
        if ext == "pdf":
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            st.session_state.pages = pdf_to_pages(tmp_path)
            os.unlink(tmp_path)
        else:
            st.session_state.pages = [Image.open(io.BytesIO(data)).convert("RGB")]
        st.session_state.page_idx = 0
        st.session_state.boxes = []
        st.query_params.clear()

    if st.session_state.pages:
        n = len(st.session_state.pages)
        if n > 1:
            c1, c2, c3 = st.columns([1,2,1])
            with c1:
                if st.button("◀ Prev") and st.session_state.page_idx > 0:
                    st.session_state.page_idx -= 1
                    st.session_state.boxes = []
                    st.query_params.clear()
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
                    st.query_params.clear()
                    st.rerun()

        if st.session_state.active_field:
            st.markdown(f'<div class="instruction-box">🖱 Draw a rectangle around the text for '
                        f'<strong>{st.session_state.active_field}</strong>, then click '
                        f'<strong>✓ Use This Selection</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="instruction-box">👈 Select a Target Field on the left, '
                        'then draw a rectangle on the document.</div>', unsafe_allow_html=True)

        page_img = st.session_state.pages[st.session_state.page_idx]
        display_img = annotate(page_img, st.session_state.boxes)
        canvas_key = f"page_{st.session_state.page_idx}"
        selection = image_selector(display_img, canvas_key)

        # Extract button (shown when selection exists and field chosen)
        if selection and st.session_state.active_field:
            x1,y1,x2,y2 = selection["x1"],selection["y1"],selection["x2"],selection["y2"]
            if st.button(f"🔍 Extract  \"{st.session_state.active_field}\"  from selection",
                          use_container_width=True, type="primary"):
                if not api_key:
                    st.error("Enter your API key first.")
                else:
                    with st.spinner(f"Extracting…"):
                        try:
                            text = extract_region(page_img, x1, y1, x2, y2,
                                                   st.session_state.active_field, api_key)
                            st.session_state.extracted[st.session_state.active_field] = text
                            st.session_state.boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
                            # Advance to next field
                            fields = st.session_state.fields
                            if st.session_state.active_field in fields:
                                idx = fields.index(st.session_state.active_field)
                                if idx + 1 < len(fields):
                                    st.session_state.active_field = fields[idx+1]
                            st.query_params.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    else:
        st.markdown("""
        <div style="border:2px dashed #26262d;border-radius:12px;padding:60px;text-align:center">
          <div style="font-size:48px">📄</div>
          <div style="color:#6b6b78;margin-top:12px">Upload a PDF or image to get started</div>
          <div style="color:#2a2a2e;font-size:12px;margin-top:6px">Supports PDF, PNG, JPG, WEBP, TIFF</div>
        </div>""", unsafe_allow_html=True)
