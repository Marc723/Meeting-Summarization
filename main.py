#how to run type = streamlit run main.py(nama file) 
#package requirement = streamlit pyaudio torch transformers faster-whisper pyannote.audio pydub moviepy ffmpeg

import pyaudio
import wave
import torch
import re
import streamlit as st
from transformers import pipeline
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.effects import normalize
from moviepy.editor import VideoFileClip
import moviepy.editor as mp
import tempfile
import os



# Fungsi untuk merekam audio
def record_audio(filename, record_seconds=10, chunk=1024, format=pyaudio.paInt16, channels=1, rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    st.write("Recording audio...")
    frames = []

    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Fungsi untuk preprocessing audio (normalisasi)
def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    normalized_audio = normalize(audio)
    processed_path = "normalized_" + os.path.basename(file_path)
    normalized_audio.export(processed_path, format="wav")
    return processed_path

# Fungsi untuk ekstraksi audio dari video
def extract_audio_from_video(video_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile("extracted_audio.wav")
    return "extracted_audio.wav"

# Fungsi untuk melakukan diarization
def diarize_audio(file_path):
    pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="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")
    diarization = pipeline(file_path)
    return diarization

# Fungsi untuk memperpanjang segmen diarization (untuk sinkronisasi yang lebih baik)
def extend_diarization_boundaries(diarization, margin=0.5):
    adjusted_diarization = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = max(0, turn.start - margin)
        end = turn.end + margin
        adjusted_diarization.append((start, end, speaker))
    return adjusted_diarization

# Fungsi untuk transkripsi audio dengan diarization yang disinkronkan
def transcribe_with_speaker(file_path, diarization):
    model = WhisperModel("medium")
    segments, _ = model.transcribe(file_path, vad_filter=True)  

    diarization_result = extend_diarization_boundaries(diarization)

    transcript = ""
    for segment in segments:
        start_time = segment.start
        end_time = segment.end
        speaker = 'unknown'

        # Mencari segmen diarization yang cocok dengan segmen transkripsi
        for turn_start, turn_end, spk in diarization_result:
            if turn_start <= end_time and start_time <= turn_end:
                speaker = spk
                break
        
        # Menyusun hasil transkripsi dengan label pembicara
        transcript += f"[{start_time:.2f} - {end_time:.2f}] Speaker {speaker}: {segment.text}\n"

    return transcript

# Fungsi untuk transkripsi audio tanpa diarization
def transcribe_audio(file_path):
    model_size = "large-v3"
    
    # Run on GPU with FP16 if available, else fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Transcribe the audio file
    segments, info = model.transcribe(file_path, beam_size=5)
    
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    # Collect transcription text
    transcript = ""
    for segment in segments:
        transcript += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
    
    return transcript

# Summarization function using Hugging Face pipeline
def summarize_text(input_text):
    device = 0 if torch.cuda.is_available() else -1  
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    # Split long text into chunks to avoid exceeding the token limit
    def split_text(text, chunk_size=1000):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    chunks = split_text(input_text)
    summaries = []
    
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    final_summary = ' '.join(summaries)
    return final_summary

# Post-processing to remove unnecessary words and clean up text
def clean_summary_text(text):
    text = re.sub(r'\b(?:um|uh|like|so|you know|well)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Convert sentences to bullet points and filter out unimportant details
def summary_to_bullets(summary_text):
    summary_text = clean_summary_text(summary_text)  
    sentences = summary_text.split('. ')  
    bullet_points = ["- " + sentence.strip() for sentence in sentences if len(sentence) > 10 and sentence]
    return '\n'.join(bullet_points)

# Streamlit app for uploading or recording and processing the file
def main():
    st.title("Audio/Video Transcription and Summarization")

    # Pilihan antara merekam, mengunggah audio, atau mengunggah video
    option = st.selectbox("Choose an option:", ["Upload Audio", "Record Audio", "Upload Video"])

    file_path = None

    if option == "Upload Audio":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if uploaded_file is not None:
            # Simpan file yang diunggah ke file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                file_path = tmp_file.name

    elif option == "Record Audio":
        record_seconds = st.slider("Select recording duration (seconds):", 5, 60, 30)
        if st.button("Start Recording"):
            # Simpan rekaman ke file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                record_audio(tmp_file.name, record_seconds=record_seconds)
                st.write(f"Recorded audio for {record_seconds} seconds.")
                file_path = tmp_file.name

    elif option == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv"])
        if uploaded_file is not None:
            # Simpan file video yang diunggah ke file sementara
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                video_file_path = tmp_file.name
            # Ekstrak audio dari video
            file_path = extract_audio_from_video(video_file_path)

    if file_path:
        # Preprocessing (opsional, jika ingin menggunakan normalisasi)
        processed_file_path = preprocess_audio(file_path)

        # Step 1: Transkripsi
        st.write("Transcribing audio...")
        transcription = transcribe_audio(processed_file_path)
        st.write("\n### Transcription Before Summarization:")
        st.text(transcription)

        # Step 2: Ringkasan
        st.write("\nSummarizing transcription...")
        summary_bullets = summary_to_bullets(summarize_text(transcription))
        st.write("\n### Final Summary After Summarization:")
        st.text(summary_bullets)

if __name__ == "__main__":
    main()
