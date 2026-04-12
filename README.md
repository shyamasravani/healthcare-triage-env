---
title: Healthcare Triage Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Healthcare Triage Environment

## 📖 Overview
This project implements a healthcare triage environment using **FastAPI** and containerized deployment on Hugging Face Spaces.  
It exposes three endpoints (`/reset`, `/step`, `/state`) that allow interaction with the environment for decision‑making workflows.

---

## ⚙️ Setup Instructions

### Local Development
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd healthcare-triage-env
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate # Linux/Mac
3. Install dependencies
   ```bash
   pip install -r requirements.txt
4. Run the app locally
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860
   
## Deployment
The app is deployed on Hugging Face Spaces using Docker.
Public Space URL:

Code
https://shyamaDatascientist-healthcare-triage-env.hf.space   

## API Endpoints
5. Reset Environment
   ```bash
   curl -X POST https://shyamaDatascientist-healthcare-triage-env.hf.space/reset
6. Step Through Environment
   ```bash
	curl -X POST https://shyamaDatascientist-healthcare-triage-env.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"decision": "Refer"}'
7. Get the current state`
   ```bash
   curl https://shyamaDatascientist-healthcare-triage-env.hf.space/state


## Inference Script
8. Run inference programmatically
python inference.py or
API_BASE_URL="https://shyamaDatascientist-healthcare-triage-env.hf.space" python inference.py

## Validation
Step 9: Confirm validation
Endpoints tested with curl and inference.py.

Container logs show successful responses (200 OK).

Manual validation script confirms /reset, /step, /state all respond correctly.

pyproject.toml included for multi‑mode deployment compliance.

Phase 1 automated checks pass with Dockerfile, inference script, and OpenEnv validation.

## Project Structure
Code
healthcare-triage-env/
├── app.py
├── inference.py
├── requirements.txt
├── Dockerfile
├── README.md
├── pyproject.toml
└── LICENSE