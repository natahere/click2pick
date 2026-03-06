---
title: DocExtract
emoji: 🔍
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🔍 DocExtract — AI Document Data Extractor

Extract structured data from any document using Claude Vision AI.

## ✨ Features

| Feature | Details |
|---|---|
| **Document types** | PDF (multi-page), PNG, JPG, WEBP, TIFF, BMP |
| **Click-to-extract** | Draw a box on any text region → instantly extracted |
| **Auto-extract all** | Claude reads the whole page and fills every field at once |
| **Templates** | Invoice, ID/Passport, Receipt, Contract, Generic Key-Value |
| **Custom fields** | Define your own list of fields |
| **Export** | JSON & CSV download |

## 🚀 Deploy to Hugging Face Spaces (FREE)

### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name it `docextract` (or anything you like)
3. Select **Gradio** as the SDK
4. Set visibility to **Public** (free) or **Private** (requires Pro)
5. Click **Create Space**

### Step 2 — Upload the files

**Option A — Git (recommended)**
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/docextract
cd docextract
# Copy app.py, requirements.txt, README.md here
git add .
git commit -m "Initial upload"
git push
```

**Option B — Web UI**  
In your Space page → Files → Upload files → drag `app.py` + `requirements.txt` + `README.md`

### Step 3 — Add your API key as a Secret (optional but recommended)

In your Space → **Settings** → **Repository secrets** → Add:

```
Name:  ANTHROPIC_API_KEY
Value: sk-ant-api03-...
```

This pre-fills the key so users don't need to enter it themselves.  
If you leave it out, users can paste their own key in the UI.

That's it — your Space builds automatically in ~2 minutes and is live at:  
`https://huggingface.co/spaces/YOUR_USERNAME/docextract`

---

## 🔑 Getting an Anthropic API Key

1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Sign up / log in → **API Keys** → **Create key**
3. Paste it into the app's **Anthropic API Key** field

---

## 🛠 How It Works

```
Upload doc → Claude Vision renders it → Select target field
→ Draw rectangle on document text → Region sent to Claude OCR
→ Extracted text populates the field → Export JSON/CSV
```

For **Auto-Extract All**: the entire page image is sent once to Claude with your field list → all fields filled in a single API call.

---

## 📁 Files

```
docextract/
├── app.py            # Gradio application
├── requirements.txt  # Python dependencies  
└── README.md         # This file (also configures the Space)
```
