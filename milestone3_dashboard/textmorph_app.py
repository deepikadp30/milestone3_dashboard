# texmorph_app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import textstat
import threading

# -------------------------------
# Backend (Flask API)
# -------------------------------
app = Flask(__name__)
CORS(app)

# Load T5 model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def paraphrase_text(text, max_length=256):
    input_text = "paraphrase: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased

def generate_summary(text):
    sentences = text.split('.')
    summary = '. '.join(sentences[:3]).strip()
    if not summary.endswith('.'):
        summary += '.'
    return summary

def calculate_readability(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
    }

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    data = request.json
    text = data.get("text", "")
    paraphrased = paraphrase_text(text)
    summary = generate_summary(paraphrased)
    readability = calculate_readability(paraphrased)
    language = "English"
    return jsonify({
        "original": text,
        "paraphrased": paraphrased,
        "summary": summary,
        "readability": readability,
        "language": language
    })

def run_flask():
    app.run(port=5000)

# -------------------------------
# Frontend (Streamlit)
# -------------------------------
st.set_page_config(page_title="Texmorph App", layout="wide")
st.title("📝 Texmorph Text Paraphrasing & Analysis")

# Start Flask in a separate thread
threading.Thread(target=run_flask, daemon=True).start()

user_text = st.text_area("Enter text to paraphrase:", height=150)

if st.button("Generate Paraphrase"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        response = paraphrase_text(user_text)
        summary = generate_summary(response)
        readability = calculate_readability(response)
        language = "English"

        st.subheader("Paraphrased Text")
        st.write(response)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Readability Scores")
        df_read = pd.DataFrame({
            "Metric": list(readability.keys()),
            "Score": list(readability.values())
        })
        st.dataframe(df_read)

        fig, ax = plt.subplots()
        ax.bar(df_read["Metric"], df_read["Score"], color=['skyblue', 'salmon', 'lightgreen'])
        ax.set_ylabel("Score")
        ax.set_title("Readability Scores")
        st.pyplot(fig)

        st.subheader("Language Detected")
        st.write(language)
