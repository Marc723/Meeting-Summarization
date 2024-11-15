import os
import wave
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from transformers import pipeline
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.effects import normalize
import uvicorn
import tempfile

# temporary file
TEMP_AUDIO_DIR = 'temp_audio'
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    normalized_audio = normalize(audio)
    processed_path = os.path.join(TEMP_AUDIO_DIR, "normalized_" + os.path.basename(file_path))
    normalized_audio.export(processed_path, format="wav")
    return processed_path

def transcribe_audio(processed_path):
    model_size = "medium"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(processed_path)
    transcription_result = {"segments": [{"text": segment.text} for segment in segments]}

    return transcription_result

def summarize_text(input_text):
    device = 0 if torch.cuda.is_available() else -1  
    summarizer = pipeline("summarization", model="bart-finetuning-samsum", device=device)

    def split_text(text, chunk_size=1000):
        # Join all text segments into a single string for chunking
        joined_text = ' '.join([t["text"] for t in text if "text" in t])
        return [joined_text[i:i + chunk_size] for i in range(0, len(joined_text), chunk_size)]
      
    chunks = split_text(input_text)
    summaries = []
    
    for chunk in chunks:
        summary = summarizer(chunk, max_length=120, min_length=30, length_penalty=0.8, no_repeat_ngram_size=2, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    final_summary = ' '.join(summaries)
    return final_summary


def summary_to_bullets(final_summary):
    sentences = final_summary.split('. ')  
    bullet_points = ["- " + sentence.strip() for sentence in sentences if len(sentence) > 10 and sentence]
    return "\n".join(bullet_points)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-audio/", response_class=HTMLResponse)
async def upload_audio(request: Request, file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file.filename.split('.')[-1]) as tmp_file:
        tmp_file.write(await file.read())
        file_path = tmp_file.name
    
    processed_file_path = preprocess_audio(file_path)
    transcription = transcribe_audio(processed_file_path)
    summary_bullets = summary_to_bullets(summarize_text(transcription["segments"]))

    return templates.TemplateResponse("result.html", {
        "request": request,
        "transcription": transcription,
        "summary": summary_bullets
    })

@app.post("/record-audio/", response_class=HTMLResponse)
async def record_audio(request: Request, audio_data: UploadFile = File(...)):
    audio_file_path = os.path.join(TEMP_AUDIO_DIR, "recorded_audio.wav")
    with open(audio_file_path, 'wb') as f:
        f.write(await audio_data.read())
    
    processed_file_path = preprocess_audio(audio_file_path)
    transcription = transcribe_audio(processed_file_path)
    summary_bullets = summary_to_bullets(summarize_text(transcription["segments"]))

    return templates.TemplateResponse("result.html", {
        "request": request,
        "transcription": transcription,
        "summary": summary_bullets
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
