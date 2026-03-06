# Minimal diagnostic app - shows exactly what's working and what's not
import streamlit as st
import io, base64, tempfile, os
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="OCR Diagnostic", layout="wide")
st.title("🔬 OCR Diagnostic")

# ── Test 1: Generate a test image with known text ─────────────────────────────
st.subheader("Step 1 — OCR Engine Status")

@st.cache_resource
def check_engines():
    results = {}
    # Tesseract
    try:
        import pytesseract
        v = pytesseract.get_tesseract_version()
        results["tesseract"] = f"✅ Available — version {v}"
        results["tesseract_obj"] = pytesseract
    except Exception as e:
        results["tesseract"] = f"❌ Not available: {e}"

    # EasyOCR
    try:
        import easyocr
        r = easyocr.Reader(['en'], gpu=False, verbose=False)
        results["easyocr"] = "✅ Available"
        results["easyocr_obj"] = r
    except Exception as e:
        results["easyocr"] = f"❌ Not available: {e}"

    # PyMuPDF
    try:
        import fitz
        results["pymupdf"] = f"✅ Available — version {fitz.version}"
    except Exception as e:
        results["pymupdf"] = f"❌ Not available: {e}"

    return results

with st.spinner("Checking engines..."):
    engines = check_engines()

st.write("**Tesseract:**", engines.get("tesseract","❓"))
st.write("**EasyOCR:**",   engines.get("easyocr","❓"))
st.write("**PyMuPDF:**",   engines.get("pymupdf","❓"))

# ── Test 2: Generate synthetic image and OCR it ───────────────────────────────
st.subheader("Step 2 — OCR on synthetic test image")

# Create a white image with black text
test_img = Image.new("RGB", (400, 100), "white")
draw = ImageDraw.Draw(test_img)
draw.text((10, 30), "Invoice Total: $1,234.56", fill="black")
st.image(test_img, caption="Test image (should say: Invoice Total: $1,234.56)", width=400)

if "tesseract_obj" in engines:
    try:
        import pytesseract
        result = pytesseract.image_to_string(test_img, config="--psm 6").strip()
        st.success(f"Tesseract OCR result: `{result}`")
    except Exception as e:
        st.error(f"Tesseract failed: {e}")

if "easyocr_obj" in engines:
    try:
        import numpy as np
        arr = np.array(test_img)
        result = " ".join(engines["easyocr_obj"].readtext(arr, detail=0))
        st.success(f"EasyOCR result: `{result}`")
    except Exception as e:
        st.error(f"EasyOCR failed: {e}")

# ── Test 3: Upload your own file ──────────────────────────────────────────────
st.subheader("Step 3 — Test with your actual file")

uploaded = st.file_uploader("Upload a PDF or image", type=["pdf","png","jpg","jpeg"])
if uploaded:
    data = uploaded.read()
    ext = uploaded.name.rsplit(".",1)[-1].lower()

    if ext == "pdf":
        import fitz
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data); path = tmp.name
        doc = fitz.open(path)
        page = doc[0]
        
        st.write("**PDF text layer (no OCR):**")
        text = page.get_text()
        st.code(text[:500] if text.strip() else "(empty — scanned PDF, needs OCR)")
        
        # Render page as image
        pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        os.unlink(path)
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")

    st.image(img, caption="Uploaded document", use_container_width=True)

    # OCR the full image
    if "tesseract_obj" in engines:
        st.write("**Tesseract on full page:**")
        with st.spinner("Running..."):
            try:
                import pytesseract
                gray = img.convert("L")
                result = pytesseract.image_to_string(gray, config="--psm 6").strip()
                st.code(result[:1000] if result else "(no text detected)")
            except Exception as e:
                st.error(f"Error: {e}")

    if "easyocr_obj" in engines:
        st.write("**EasyOCR on full page:**")
        with st.spinner("Running (may take 30s)..."):
            try:
                import numpy as np
                arr = np.array(img)
                result = "\n".join(engines["easyocr_obj"].readtext(arr, detail=0))
                st.code(result[:1000] if result else "(no text detected)")
            except Exception as e:
                st.error(f"Error: {e}")
