import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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

# --- 1. CONFIGURATION ---
BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

# ADAPTER_PATH = r"model/sft"

# NEW LINE (Update to your new model repo):
ADAPTER_PATH = "gokouming/noval-adapter"

# --- 2. LOAD MODEL ---
print("--- STARTING APP ---")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("✅ Adapter loaded.")
    except:
        print("⚠️ Adapter not found, using base model.")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
except Exception as e:
    print(f"❌ Model Error: {e}")


# --- 3. PDF GENERATION ---
# --- 3. PDF GENERATION (CHINESE) ---
def create_pdf_report(image_path, process_text):
    output_filename = "Production_Process_Report.pdf"

    # Register Chinese Font (SimHei is standard on Windows)
    try:
        pdfmetrics.registerFont(TTFont('SimHei', 'C:\\Windows\\Fonts\\simhei.ttf'))
        font_name = 'SimHei'
    except:
        print("⚠️ SimHei font not found. Chinese characters may not render.")
        font_name = 'Helvetica'

    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # Styles using the Chinese font
    title_style = ParagraphStyle('ChineseTitle', parent=styles['Title'], fontName=font_name, fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle('ChineseHeading', parent=styles['Heading2'], fontName=font_name, fontSize=14,
                                   spaceAfter=10)
    body_style = ParagraphStyle('ChineseBody', parent=styles['BodyText'], fontName=font_name, fontSize=12, leading=18)

    # CHINESE TITLE
    story.append(Paragraph("Production Report", title_style))
    story.append(Spacer(1, 0.5 * inch))

    if image_path:
        img = ReportLabImage(image_path, width=6 * inch, height=4 * inch, kind='proportional')
        story.append(img)
        story.append(Spacer(1, 0.5 * inch))

    # CHINESE HEADING
    story.append(Paragraph("Detected Workflow:", heading_style))

    steps = process_text.replace("-", "\n• ").split("\n")
    for step in steps:
        story.append(Paragraph(step, body_style))
    doc.build(story)
    return output_filename

# --- 4. AI INFERENCE (CHINESE PROMPT) ---
def analyze_and_generate_report(image_path):
    if not image_path: return "Please upload an image.", None

    # CHANGED PROMPT TO CHINESE
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Please analyze the product image and generate a detailed step-by-step manufacturing process workflow in English."}
            ]
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True,
                       return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = \
    processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    pdf_path = create_pdf_report(image_path, output_text)
    return "", pdf_path


# --- 5. HELPER: FIX PATHS IN HTML ---
def prepare_html(filename):
    file_path = f"html/{filename}"
    if not os.path.exists(file_path): return f"<h1>Error: {filename} not found</h1>"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. FIX THE CSS FILENAME TYPO (Critical Fix)
    # Changes "styles.css" (plural) to "style.css" (singular) to match your folder
    content = content.replace('styles.css', 'style.css')

    # 2. Fix CSS/Image Paths
    content = content.replace('href="../css/', 'href="/static/css/')
    content = content.replace('href="css/', 'href="/static/css/')
    content = content.replace('src="../img/', 'src="/static/img/')
    content = content.replace('src="img/', 'src="/static/img/')
    content = content.replace('href="../img/', 'href="/static/img/')
    content = content.replace('href="img/', 'href="/static/img/')

    # 3. Fix Page Links
    url_map = {
        "index.html": "/", "products.html": "/products", "solutions.html": "/solutions",
        "research.html": "/research", "price.html": "/price", "contact.html": "/contact",
        "about.html": "/about", "blog.html": "/blog"
    }
    # Replace every variation of the link (e.g., "html/about.html", "../about.html", "about.html")
    for page, route in url_map.items():
        content = content.replace(f'href="{page}"', f'href="{route}"')
        content = content.replace(f'href="html/{page}"', f'href="{route}"')
        content = content.replace(f'href="../{page}"', f'href="{route}"')
        content = content.replace(f'href="../html/{page}"', f'href="{route}"')

    return content


# --- 6. GRADIO INTERFACE (THE TOOL) ---
custom_css = """
footer { visibility: hidden !important; }
.gradio-container { background-color: #0b0b0f !important; margin: 0 !important; max-width: 100% !important; min-height: 100vh; }
#tool-container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }

/* White Background for Output Box */

/* FIX: COMPLETELY LOCK THE TEXTBOX RESIZING */
#analysis-output textarea {
    background-color: #e5e7eb !important; 
    color: #555555 !important; 
    font-size: 1rem !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    cursor: not-allowed !important;
    resize: none !important;
    min-height: 300px !important;
    max-height: 300px !important;
    text-align: center !important; /* Centers the locked message */
    padding-top: 120px !important; /* Pushes text to middle vertically */
}

/* Label Styling */
label span { color: #9ca3af !important; font-weight: bold; }

/* Header Styles */
.tool-header { text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #27272a; }
.tool-logo-text { font-size: 2.5rem; font-weight: 700; color: white; letter-spacing: -1px; }
.tool-logo-text span { color: #0262F2; }
.tool-tagline { color: #9ca3af; font-size: 1rem; margin-top: 5px; }

/* Button Style */
button.primary { background-color: #0262F2 !important; color: white !important; border: none !important; font-weight: 600 !important; }
h1, h2 { color: white !important; font-family: sans-serif; }
"""

with gr.Blocks(title="Noval AI Tool",
               css=custom_css,
               theme=None,
                head='<link rel="icon" href="/static/img/OIP.png" type="image/png">' # <--- THIS SETS THE ICON
    ) as demo:
    gr.HTML(
        '<div style="padding: 20px;"><a href="/" style="color: #9ca3af; text-decoration: none; font-family: sans-serif;">&larr; Back to Website</a></div>')

    with gr.Group(elem_id="tool-container"):
        # --- HEADER ---
        gr.HTML("""
            <div class="tool-header">
                <div class="tool-logo-text">noval<span>.ai</span></div>
                <div class="tool-tagline">The AI Cloud for Manufacturing</div>
            </div>
        """)

        gr.Markdown("# AI Process Generator")
        gr.Markdown("Upload a product photo below to generate a technical workflow report.")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="filepath", height=400, label="Upload Image")
                btn = gr.Button("Generate Report 🚀", variant="primary", size="lg")
            with gr.Column(scale=1):
                # ID allows CSS targeting
                out_text = gr.Textbox(
                    label="Fine-Tuning Parameters",
                    lines=10,
                    show_copy_button=False,
                    elem_id="analysis-output",
                    placeholder="🔒 Configure Fine-Tuning Parameters here...\n(Upgrade to Membership to customize fine-tuning results)",
                    interactive=False
                )
                out_file = gr.File(label="Download PDF Report", file_types=[".pdf"])
        btn.click(analyze_and_generate_report, inputs=input_img, outputs=[out_text, out_file])

# --- 7. FASTAPI SERVER ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(): return prepare_html("index.html")


@app.get("/products", response_class=HTMLResponse)
async def products(): return prepare_html("products.html")

@app.get("/solutions", response_class=HTMLResponse)
async def solutions(): return prepare_html("solutions.html")

@app.get("/research", response_class=HTMLResponse)
async def research(): return prepare_html("research.html")

@app.get("/price", response_class=HTMLResponse)
async def price(): return prepare_html("price.html")

@app.get("/contact", response_class=HTMLResponse)
async def contact(): return prepare_html("contact.html")

@app.get("/about", response_class=HTMLResponse)
async def about(): return prepare_html("about.html")  # Fixed function name

@app.get("/blog", response_class=HTMLResponse)
async def blog(): return prepare_html("blog.html")    # Fixed function name


app = gr.mount_gradio_app(app, demo, path="/tool", root_path="/tool")


if __name__ == "__main__":
    print("👉 WEBSITE: http://127.0.0.1:7861")
    uvicorn.run(app, host="0.0.0.0", port=7861)
