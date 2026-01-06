import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import torch
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- CONFIGURATION ---
BASE_MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
ADAPTER_PATH = "gokouming/noval-adapter"

# Get Hugging Face token from environment (if adapter is private)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# --- LOAD MODEL ---
print("--- STARTING APP ---")
try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # ← Changed from "cpu"
        trust_remote_code=True,
        token=HF_TOKEN  # ← Added for private models
    )
    try:
        model = PeftModel.from_pretrained(
            model, 
            ADAPTER_PATH,
            token=HF_TOKEN  # ← Added for private adapters
        )
        print("✅ Adapter loaded.")
    except:
        print("⚠️ Adapter not found, using base model.")
    
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        token=HF_TOKEN
    )
except Exception as e:
    print(f"❌ Model Error: {e}")

# ... rest of your code stays the same ...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # ← Railway provides PORT
    print(f"👉 WEBSITE: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
