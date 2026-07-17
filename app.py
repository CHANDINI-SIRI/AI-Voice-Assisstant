import os
import tempfile
import gradio as gr
from google import genai
import speech_recognition as sr
from gtts import gTTS


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def speech_to_text(audio):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio_data)
    except Exception:
        return "Sorry, I couldn't understand."

def ask_gemini(text):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text
    )
    return response.text


def text_to_speech(text):
    audio_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    ).name

    gTTS(text=text).save(audio_path)

    return audio_path


def voice_assistant(audio):
    if audio is None:
        return "Please upload or record audio.", "", None

    user_text = speech_to_text(audio)
    ai_reply = ask_gemini(user_text)
    voice = text_to_speech(ai_reply)

    return user_text, ai_reply, voice


demo = gr.Interface(
    fn=voice_assistant,
    inputs=gr.Audio(
        type="filepath",
        label="Speak"
    ),
    outputs=[
        gr.Textbox(label="You Said"),
        gr.Textbox(label="Gemini Response"),
        gr.Audio(label="Voice Response")
    ],
    title="AI Voice Assistant",
    description="Speak to Gemini and receive both text and voice responses."
)


port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
