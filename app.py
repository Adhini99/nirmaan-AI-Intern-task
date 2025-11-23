# app placeholder
# app.py
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from scorer import RubricScorer
import uvicorn
import json

RUBRIC_PATH = "Case study for interns.xlsx"  # replace if needed

app = FastAPI()
scorer = RubricScorer(RUBRIC_PATH)

class TranscriptIn(BaseModel):
    transcript: str

@app.post("/score")
async def score_endpoint(payload: TranscriptIn):
    transcript = payload.transcript
    res = scorer.compute_scores(transcript)
    return res

@app.post("/score-file")
async def score_file(file: UploadFile = File(...)):
    content = (await file.read()).decode('utf-8')
    res = scorer.compute_scores(content)
    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
