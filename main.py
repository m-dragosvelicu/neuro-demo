from audio_recorder_streamlit import audio_recorder
import streamlit as st
import whisper
import numpy as np
import time
import pandas as pd
import altair as alt

# Load Whisper Tiny once at startup
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()

st.title("🧠 AI Language Visualizer (Demo)")
# Language selector
lang_display = st.radio(
    "Alege limba:",
    ("English"), # , "Romana"),
    horizontal=True,
    index=0,
)
lang_code = "en" if lang_display == "English" else "ro"
st.write("Spune ceva in microfon. Vezi cum este spart in tokens!")

# Step 1: Record audio
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # Save file to disk
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    result = model.transcribe(
        "temp.wav",
        language=lang_code,
        word_timestamps=True
    )
    st.write(result["text"])

    # Visualise tokens as a heat‑map
    chart_data = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            # Whisper ≤2025‑05 uses key "word"; newer nightlies use "text"
            token = w.get("word") or w.get("text") or w.get("token") or ""
            chart_data.append(
                {"token": token.strip(), "start": w["start"], "end": w["end"]}
            )

    if chart_data:
        df = pd.DataFrame(chart_data)
        heatmap = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x='start:Q',
                x2='end:Q',
                y='token:N',
                color='start:Q'
            )
            .properties(height=400)
        )
        st.altair_chart(heatmap, use_container_width=True)
else:
    st.info("Click to record and say something!")