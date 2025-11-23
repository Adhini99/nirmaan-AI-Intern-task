# Case Study Repo
Spoken Introduction Scoring Tool – Case Study Submission

This repository contains a small end-to-end system that evaluates a student’s spoken introduction using a rubric and a transcript.
It was created as part of the Nirmaan AI Engineer Intern – Case Study.

The tool takes a transcript, compares it against the rubric criteria using rule-based + semantic + length-based scoring, and returns:

Overall score (0–100)

Per-criterion score

Keyword matches

Semantic similarity

Feedback for improvement

🚀 Features

✔️ Accepts raw transcript text or uploaded .txt file

✔️ Reads rubric from Excel (case_study_rubric.xlsx)

✔️ Uses sentence-transformers for semantic similarity

✔️ Explainable scoring (keyword match, length check, similarity score)

✔️ REST API using FastAPI

✔️ Optional UI using Streamlit

✔️ Fully runnable locally with simple commands

📊 Scoring Formula

Each rubric criterion is evaluated using three signals:

1. Rule-Based Score (S_rule)

Checks presence of rubric keywords in transcript.

S_rule = (# matched keywords) / (total keywords)

2. Semantic Score (S_sem)

Cosine similarity between:

embedding(criterion_description)

embedding(transcript)

Mapped to 0–1:

S_sem = (cosine_similarity + 1) / 2

3. Length Score (S_len)

If rubric specifies min/max words:

If within range → S_len = 1
Else penalty proportional to difference

Combined Score
S_combined = 0.4*S_rule + 0.5*S_sem + 0.1*S_len


Each rubric criterion has a weight → normalized.

Final Overall Score
overall_raw = Σ (S_combined_i × normalized_weight_i)
overall_percentage = overall_raw × 100

📁 Repository Structure
├── app.py                # FastAPI backend
├── scorer.py             # Rubric parser + scoring logic
├── streamlit_app.py      # UI (optional)
├── case_study_rubric.xlsx  # Rubric file
├── requirements.txt
├── README.md
└── LOCAL_RUN.md          # Step-by-step run instructions

🧪 Run Locally (Quick Version)
1. Create a virtualenv
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

2. Install requirements
pip install -r requirements.txt

3. Run backend
uvicorn app:app --reload --port 8000


Open API docs:
👉 http://localhost:8000/docs

4. (Optional UI) Run Streamlit
streamlit run streamlit_app.py


Open UI:
👉 http://localhost:8501

📄 Rubric File Requirements

Your Excel should contain columns:

criterion_id

criterion_name

description

keywords

weight

min_words

max_words

Rename your rubric file to:

case_study_rubric.xlsx


Place it in repo root.

📝 Deployment

Deployment is optional for this task.
This submission includes all required components:

Source code

requirements.txt

README with scoring formula

Document with local-run steps

🎥 Suggested Screen Recording

Record a short video showing:

Run backend (uvicorn)

Open docs in browser

Run Streamlit UI

Copy/paste transcript → Score → Show output