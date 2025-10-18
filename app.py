import streamlit as st
import os
import tempfile
import subprocess
import cv2
import speech_recognition as sr
import google.generativeai as genai
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

# Gemini integration and search tool
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')
search_tool = DuckDuckGoSearchRun()

# Streamlit setup
st.set_page_config(page_title="AI Multimodal Analyzer", layout="wide")
st.title("🧠 AI Multimodal Analyzer")
st.markdown("Analyze **images, videos, or perform web searches** using Gemini AI.")

# Temporary file handler
def save_temp_file(uploaded_file):
    if uploaded_file:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            return tmp.name
    return None

# Extract first frame from video
def extract_video_thumbnail(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        thumb_path = video_path + "_thumb.jpg"
        cv2.imwrite(thumb_path, frame)
        return thumb_path
    return None

# Extract & transcribe audio from video
def transcribe_audio(video_path):   
    audio_path = video_path + "_audio.wav"
    subprocess.run(
        ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    recognizer = sr.Recognizer()
    transcription = ""
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio)
        except Exception as e:
            transcription = f"Transcription error: {e}"
    os.remove(audio_path)
    return transcription

# Gemini content analysis
def analyze_content(prompt, image_path=None, transcription=None):
    content = prompt
    if transcription:
        content += f"\n\nTranscription:\n{transcription}"
    
    try:
        if image_path:
            with open(image_path, "rb") as img:
                response = model.generate_content([
                    content,
                    {"mime_type": "image/jpeg", "data": img.read()}
                ])
        else:
            response = model.generate_content(content)
        return response.text
    except Exception as e:
        return f"Error during analysis: {e}"

# Web search
def perform_web_search(query):
    results = search_tool.run(query)
    prompt = f"Web search results for '{query}':\n{results}\n\nProvide a concise AI-powered summary and analysis."
    return analyze_content(prompt)

# Sidebar mode
analyze_type = st.sidebar.radio("Choose analysis type:", ['Image', 'Video', 'Web Search'])

# IMAGE MODE
if analyze_type == "Image":
    image_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if st.button("Analyze Image") and image_file:
        image_path = save_temp_file(image_file)
        st.image(image_path, width=400)
        prompt = "Provide a detailed and structured analysis of this image."
        result = analyze_content(prompt, image_path=image_path)
        st.markdown(result)
        os.remove(image_path)

#VIDEO MODE
elif analyze_type == "Video":
    video_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
    if st.button("Analyze Video") and video_file:
        video_path = save_temp_file(video_file)
        st.video(video_path)

        st.info("🔍 Extracting thumbnail and transcribing audio...")
        thumb_path = extract_video_thumbnail(video_path)
        transcription = transcribe_audio(video_path)

        if thumb_path:
            st.image(thumb_path, caption="Extracted Thumbnail", width=400)
        
        prompt = "Analyze this video based on its visual and audio content. Summarize key themes or actions."
        result = analyze_content(prompt, image_path=thumb_path, transcription=transcription)

        st.markdown("### 🎯 Gemini AI Analysis:")
        st.write(result)

        # Cleanup
        os.remove(video_path)
        if thumb_path:
            os.remove(thumb_path)

#WEB SEARCH MODE
elif analyze_type == "Web Search":
    query = st.text_input("Enter your search query:")
    if st.button("Search and Analyze") and query:
        st.info("🔎 Performing web search and generating analysis...")
        result = perform_web_search(query)
        st.markdown("### 🧭 AI Web Analysis:")
        st.write(result)