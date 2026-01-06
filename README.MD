---
title: Noval AI Manufacturing Intelligence
emoji: 🏭
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered manufacturing process report generator.
python_version: "3.10"
fullWidth: true
models:
  - Qwen/Qwen2.5-VL-7B-Instruct
  - gokouming/noval-adapter
---

# 🏭 Noval AI - Manufacturing Intelligence Platform

![Noval AI Banner](img/OIP.png)

**Noval AI** is an AI-native cloud platform designed for the manufacturing industry. It leverages state-of-the-art **Vision-Language Models (Qwen2.5-VL)** to analyze product images and automatically generate technical manufacturing process reports.

The project combines a modern marketing website (FastAPI + HTML/CSS) with a powerful interactive AI tool (Gradio).

---

## 🚀 Key Features

* **AI Process Generation:** Upload a product image (e.g., CNC part, Injection Mold) and receive a detailed step-by-step manufacturing workflow.
* **Vision-Language Intelligence:** Powered by **Qwen2.5-VL-7B-Instruct** fine-tuned with **LoRA** for industrial contexts.
* **Automated PDF Reporting:** Instantly generates professional PDF reports using `ReportLab`.
* **Full-Stack Web Platform:** Includes a complete marketing frontend (Home, Solutions, Pricing, Research) served via FastAPI.
* **Interactive Tool:** Seamlessly integrated Gradio interface available at `/tool`.

---

## 🛠️ Tech Stack

* **Core AI:** [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers/index), [PEFT](https://github.com/huggingface/peft)
* **Backend:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
* **Frontend UI:** [Gradio](https://gradio.app/), HTML5, CSS3
* **Deployment:** Hugging Face Spaces (Gradio SDK)

---

## ⚡ Local Installation

### Prerequisites
* **GPU:** NVIDIA RTX 3090 Ti (24GB VRAM) or equivalent recommended.
* **Python:** 3.10+

### 1. Clone the Repository
```bash
git clone [https://github.com/SuperGokou/noval-ai.git](https://github.com/SuperGokou/noval-ai.git)
cd noval-ai