SKILL LENS AI - Resume Screener
Tagline: “Skip the noise, meet the talent.”

Features
Upload resumes in PDF format.
Extract text from resumes and perform semantic search using Sentence Transformers.
Compare candidate skills with job descriptions to calculate similarity scores.
Dashboard built with Streamlit to display candidate rankings, scores, and resume previews.
Optionally link and verify candidates’ profiles from GitHub and LinkedIn.
Detect potentially fake or misleading resumes (fraud detection module).
Progress visualization while processing multiple resumes.
Tech Stack
Backend: Python, PyTorch, Sentence Transformers
Frontend: Streamlit, HTML/CSS (for dashboard customization)
PDF Handling: PyPDF2
Libraries: Pandas, io, time, util (from sentence_transformers)
Installation
Clone the repository:
git clone <your-repo-link>
cd Nightangle
Create a virtual environment (Linux example):
python -m venv venv
source venv/bin/activate
Install required packages:
pip install -r requirements.txt
Running the App
streamlit run app.py
How It Works (Step by Step)
Upload a PDF resume.
The system extracts text from the PDF using PyPDF2.
Resume text is encoded into embeddings using a pretrained MiniLM model.
Compare candidate embeddings with the job description embedding to calculate similarity.
Display candidates in the dashboard with similarity scores and preview of their resume.
Optional GitHub/LinkedIn verification for authenticity.
Future Enhancements
Add multi-language support for resumes.

Integrate an interactive feedback system for recruiters.

Expand fraud detection using AI-powered anomaly detection.

Deploy on cloud services for enterprise usage.

Shivam Jain – Project Lead & Backend Developer
