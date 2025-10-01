import streamlit as st
import pandas as pd
import time
import PyPDF2
import pdfplumber
import textstat
from readability import Readability
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from textblob import TextBlob
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import numpy as np
import difflib
import io

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Milestone 3 Dashboard (Merged)", layout="wide")
st.title("📊 Milestone 3 Dashboard — Training, Summaries & Texmorph")

# -------------------------------
# Utilities: text extraction & helpers
# -------------------------------
def extract_text_from_pdf_filelike(filelike):
    # Try pdfplumber first for better extraction
    try:
        text = ""
        with pdfplumber.open(filelike) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text.strip()
    except Exception:
        # fallback to PyPDF2
        try:
            filelike.seek(0)
            reader = PyPDF2.PdfReader(filelike)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + " "
            return text.strip()
        except Exception:
            return ""

def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        # pdfplumber wants a file-like object
        uploaded_file.seek(0)
        return extract_text_from_pdf_filelike(uploaded_file)
    elif name.endswith('.csv'):
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        # join all columns into a single long text
        text = " ".join(df.astype(str).apply(lambda row: " ".join(row.values.astype(str)), axis=1).tolist())
        return text
    else:
        return ""

def safe_summarizer(text, summarizer, max_chars=1000):
    if not text or not text.strip():
        return ""
    snippet = text[:max_chars]
    try:
        out = summarizer(snippet, max_length=120, min_length=30, do_sample=False)
        return out[0]['summary_text']
    except Exception:
        return ""

def calc_readability_scores(text):
    if not text or not text.strip():
        return {"Flesch": 0.0, "GunningFog": 0.0}
    try:
        r = Readability(text)
        flesch = max(min(r.flesch().score, 120), 0)
        gunning = max(min(r.gunning_fog().score, 30), 0)
        return {"Flesch": flesch, "GunningFog": gunning}
    except Exception:
        # fallback to textstat
        try:
            flesch = textstat.flesch_reading_ease(text)
            return {"Flesch": flesch, "GunningFog": 0.0}
        except Exception:
            return {"Flesch": 0.0, "GunningFog": 0.0}

def jaccard_similarity(a, b):
    wa = set(a.split())
    wb = set(b.split())
    inter = wa.intersection(wb)
    union = wa.union(wb)
    return len(inter)/len(union) if union else 0.0

def bleu_score(reference, candidate):
    try:
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    except Exception:
        return 0.0

def rouge_l_f(reference, candidate):
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(reference, candidate)
        return score['rougeL'].fmeasure
    except Exception:
        return 0.0

def highlight_word_level(ref, gen):
    """
    Returns HTML showing word-level diffs:
     - removed words in red (wrapped with <span style='color:#c00'>-word</span>)
     - added words in green (wrapped with <span style='color:#0a0'>+word</span>)
     - unchanged words as normal
    This produces a readable in-line diff of generated vs reference.
    """
    ref_words = ref.split()
    gen_words = gen.split()
    sm = difflib.SequenceMatcher(a=ref_words, b=gen_words)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            out.append(" ".join(ref_words[i1:i2]))
        elif tag == 'delete':
            # removed from ref
            removed = " ".join(ref_words[i1:i2])
            out.append(f"<span style='background:#ffd6d6;color:#700;padding:2px;border-radius:3px'>-{removed}</span>")
        elif tag == 'insert':
            added = " ".join(gen_words[j1:j2])
            out.append(f"<span style='background:#d6ffd6;color:#070;padding:2px;border-radius:3px'>+{added}</span>")
        elif tag == 'replace':
            removed = " ".join(ref_words[i1:i2])
            added = " ".join(gen_words[j1:j2])
            out.append(f"<span style='background:#ffd6d6;color:#700;padding:2px;border-radius:3px'>-{removed}</span>")
            out.append(f"<span style='background:#d6ffd6;color:#070;padding:2px;border-radius:3px'>+{added}</span>")
    return " ".join(out)

# -------------------------------
# Load models (lazy) & caching
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_paraphrase_model():
    try:
        tname = "t5-small"
        tok = T5Tokenizer.from_pretrained(tname)
        mod = T5ForConditionalGeneration.from_pretrained(tname)
        return tok, mod
    except Exception:
        return None, None

summarizer = load_summarizer_model()
paraphrase_tok, paraphrase_model = load_paraphrase_model()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Model Training", "Summarization & Evaluation", "Paraphrasing & Comparison"])

# =========================
# TAB 1: Model Training
# =========================
with tab1:
    st.header("🛠 Model Training (TF-IDF + LogisticRegression)")

    uploaded_train = st.file_uploader("Upload CSV for training", type=["csv"], key="train_upload")
    if uploaded_train:
        try:
            df_train = pd.read_csv(uploaded_train)
            st.subheader("Dataset Preview")
            st.dataframe(df_train.head())
            source_col = st.selectbox("Select text column (features)", df_train.columns, key="source_col")
            target_col = st.selectbox("Select target column (labels)", df_train.columns, key="target_col")
            test_size = st.slider("Test set fraction", 0.05, 0.5, 0.2)
            if st.button("Train Model"):
                progress = st.progress(0)
                log = st.empty()
                logs = []
                def log_update(msg, p):
                    logs.append(msg)
                    log.text("\n".join(logs))
                    progress.progress(p)
                    time.sleep(0.2)
                try:
                    log_update("Splitting dataset...", 10)
                    X = df_train[source_col].fillna("").astype(str)
                    y = df_train[target_col].fillna("").astype(str)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    log_update("Vectorizing (TF-IDF)...", 40)
                    vectorizer = TfidfVectorizer(max_features=5000)
                    X_train_vec = vectorizer.fit_transform(X_train)
                    X_test_vec = vectorizer.transform(X_test)
                    log_update("Training Logistic Regression...", 70)
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train_vec, y_train)
                    log_update("Evaluating...", 90)
                    y_pred = model.predict(X_test_vec)
                    acc = accuracy_score(y_test, y_pred)
                    joblib.dump({'model': model, 'vectorizer': vectorizer}, 'trained_model.pkl')
                    log_update(f"Model saved to trained_model.pkl (accuracy={acc:.4f})", 100)
                    st.success(f"Training complete — Accuracy: {acc*100:.2f}%")
                except Exception as e:
                    st.error(f"Training failed: {e}")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# =========================
# TAB 2: Summarization & Evaluation
# =========================
with tab2:
    st.header("📝 Summarization & Evaluation")

    uploaded_eval = st.file_uploader("Upload CSV (Original & Reference) or PDF", type=["csv","pdf"], key="summ_upload")
    if uploaded_eval:
        original_texts = []
        reference_texts = []

        if uploaded_eval.name.lower().endswith(".csv"):
            df2 = pd.read_csv(uploaded_eval)
            st.subheader("CSV Preview")
            st.dataframe(df2.head())
            original_col = st.selectbox("Select Original Text Column", df2.columns, key="orig_col")
            reference_col = st.selectbox("Select Reference Summary Column (if none, leave blank)", [""] + list(df2.columns), key="ref_col")
            original_texts = df2[original_col].fillna("").astype(str).tolist()
            if reference_col and reference_col != "":
                reference_texts = df2[reference_col].fillna("").astype(str).tolist()
            else:
                reference_texts = [""] * len(original_texts)
        else:
            # PDF
            uploaded_eval.seek(0)
            text = extract_text_from_pdf_filelike(uploaded_eval)
            if text:
                # naive split pages/paragraphs
                original_texts = [text]
                reference_texts = [""]
            else:
                st.warning("No text found in PDF.")

        if len(original_texts) == 0:
            st.warning("No usable text found in upload.")
        else:
            st.subheader("Generating summaries (transformer model)...")
            if summarizer is None:
                st.warning("Summarizer model not available. Install 'transformers' and the model.")
                generated_summaries = [""] * len(original_texts)
            else:
                generated_summaries = []
                MAX_CHARS = st.number_input("Max characters to use from original for summarization (per item)", 200, 5000, 1000)
                for txt in original_texts:
                    s = safe_summarizer(txt, summarizer, max_chars=MAX_CHARS)
                    generated_summaries.append(s if s else "Summary generation failed")

            # Comparison table (kept for dataset view)
            comparison_df = pd.DataFrame({
                "Original Text": original_texts,
                "Generated Summary": generated_summaries,
                "Reference Summary": reference_texts
            })
            st.subheader("Comparison Table (first 10 rows)")
            st.dataframe(comparison_df.head(10))

            # Compute ROUGE metrics averages
            scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
            rouge1_list, rouge2_list, rougeL_list = [], [], []
            for gen, ref in zip(generated_summaries, reference_texts):
                if not ref.strip():
                    rouge1_list.append(0.0); rouge2_list.append(0.0); rougeL_list.append(0.0)
                else:
                    scores = scorer.score(ref, gen)
                    rouge1_list.append(scores['rouge1'].fmeasure)
                    rouge2_list.append(scores['rouge2'].fmeasure)
                    rougeL_list.append(scores['rougeL'].fmeasure)

            avg_metrics = [np.mean(rouge1_list), np.mean(rouge2_list), np.mean(rougeL_list)]
            labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']

            # ROUGE radar and readability side by side
            col_r1, col_r2 = st.columns([1,1])

            with col_r1:
                st.subheader("ROUGE Radar (average)")
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
                metrics = avg_metrics + avg_metrics[:1]
                angles += angles[:1]
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(111, polar=True)
                ax.plot(angles, metrics, 'o-', linewidth=2)
                ax.fill(angles, metrics, alpha=0.25)
                ax.set_thetagrids(np.degrees(angles[:-1]), labels)
                ax.set_ylim(0,1)
                st.pyplot(fig)

            with col_r2:
                st.subheader("Readability of Generated Summaries")
                flesch_list, gunning_list = [], []
                for s in generated_summaries:
                    if not s.strip():
                        flesch_list.append(0.0); gunning_list.append(0.0)
                    else:
                        rs = calc_readability_scores(s)
                        flesch_list.append(rs["Flesch"])
                        gunning_list.append(rs["GunningFog"])
                x = np.arange(len(generated_summaries))
                fig2, ax2 = plt.subplots(figsize=(6,4))
                width = 0.35
                ax2.bar(x - width/2, flesch_list, width, label='Flesch')
                ax2.bar(x + width/2, gunning_list, width, label='Gunning Fog')
                ax2.set_xticks(x)
                labels_x = [f"S{i+1}" for i in range(len(generated_summaries))]
                ax2.set_xticklabels(labels_x, rotation=30)
                ax2.set_ylim(0, max(max(flesch_list)+1, max(gunning_list)+1, 10))
                ax2.set_ylabel("Score")
                ax2.legend()
                st.pyplot(fig2)

            # Translation of generated summaries
            st.subheader("🌐 Translate Generated Summaries")
            languages_map = {
                "English": "en", "Spanish": "es", "French": "fr", "German": "de",
                "Chinese": "zh-cn", "Hindi": "hi", "Tamil": "ta", "Japanese": "ja",
                "Russian": "ru", "Arabic": "ar"
            }
            target_lang_name = st.selectbox("Select language to translate summaries", list(languages_map.keys()), key="summ_translate_lang")
            target_code = languages_map[target_lang_name]
            translated_summaries = []
            for s in generated_summaries:
                if not s.strip():
                    translated_summaries.append("")
                else:
                    try:
                        blob = TextBlob(s)
                        translated_summaries.append(str(blob.translate(to=target_code)))
                    except Exception:
                        # fallback: no translation
                        translated_summaries.append("Translation failed")

            st.subheader(f"Translated Summaries ({target_lang_name})")
            for i, ts in enumerate(translated_summaries):
                st.markdown(f"*Summary {i+1}:* {ts}")

            # Download result
            final_df = comparison_df.copy()
            final_df["Translated Summary"] = translated_summaries
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download summaries as CSV", data=csv, file_name="summaries.csv", mime="text/csv")

# =========================
# TAB 3: Paraphrasing & Comparison (Texmorph)
# =========================
with tab3:
    st.header("✍ Paraphrase, Translate & Compare (Texmorph)")

    upl = st.file_uploader("Upload CSV or PDF to process (Input text)", type=['csv','pdf'], key="paraphrase_file")
    if upl:
        input_text = read_uploaded_file(upl)
        if not input_text:
            st.warning("No text found in uploaded file.")
            st.stop()

        st.markdown("### Original / Input Text (extracted)")
        st.write(input_text)

        # generate paraphrase (T5) if possible
        st.markdown("Generating paraphrase (T5-small)...")
        if paraphrase_tok is None or paraphrase_model is None:
            st.warning("Paraphrase model not available. Install 'transformers' and model weights.")
            generated_text = ""
        else:
            try:
                # prepare input for T5 in reasonable chunk - for long texts, shorten
                input_for_paraphrase = input_text[:4000]
                inpt = "paraphrase: " + input_for_paraphrase
                inputs = paraphrase_tok.encode(inpt, return_tensors="pt", max_length=512, truncation=True)
                outputs = paraphrase_model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
                generated_text = paraphrase_tok.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                st.error(f"Paraphrasing failed: {e}")
                generated_text = ""

        st.markdown("### Generated Text (paraphrased)")
        st.write(generated_text if generated_text else "—")

        # Auto-generate reference summary (extractive via transformer summarizer if available)
        st.markdown("### Auto-generated Reference (you may override below)")
        auto_reference = ""
        if summarizer is not None:
            auto_reference = safe_summarizer(input_text, summarizer, max_chars=2000)
        else:
            # fallback: take first 3 sentences
            auto_reference = ". ".join(input_text.split(".")[:3]).strip()
            if not auto_reference.endswith("."):
                auto_reference += "."

        # allow user to override reference text (important)
        reference_text = st.text_area("Reference Text (edit if you have a gold reference)", value=auto_reference, height=150)

        # Language selection (dynamic translation) — changing the selection re-renders translated text
        languages = {
            "English": "en", "Hindi": "hi", "French": "fr", "Spanish": "es",
            "German": "de", "Japanese": "ja", "Chinese": "zh-cn", "Arabic": "ar", "Russian": "ru"
        }
        lang_choice = st.selectbox("Select target language for translation (dynamic)", list(languages.keys()), key="paraphrase_translate")
        lang_code = languages[lang_choice]

        # automatic translation of generated text
        try:
            translated_text = TextBlob(generated_text).translate(to=lang_code) if generated_text.strip() else ""
            translated_text = str(translated_text)
        except Exception:
            # fallback to original generated text if translation failed
            translated_text = generated_text

        # Summary for generated
        gen_summary = safe_summarizer(generated_text, summarizer, max_chars=1000) if summarizer else ". ".join(generated_text.split(".")[:3])

        # Readability for input & generated
        rd_input = calc_readability_scores(input_text)
        rd_generated = calc_readability_scores(generated_text)

        # Roughness / comparison metrics using Reference (the reference_text user may have edited)
        cos_sim = 0.0
        jacc_sim = 0.0
        bleu_v = 0.0
        rouge_l_v = 0.0
        try:
            cos_sim = cosine_similarity(TfidfVectorizer().fit_transform([reference_text, generated_text]).toarray())[0,1]
        except Exception:
            cos_sim = 0.0
        jacc_sim = jaccard_similarity(reference_text, generated_text)
        bleu_v = bleu_score(reference_text, generated_text)
        rouge_l_v = rouge_l_f(reference_text, generated_text)

        # -------------------------------
        # Display horizontal side-by-side paragraph boxes
        # -------------------------------
        st.subheader("Side-by-Side Comparison (Input → Generated → Reference)")

        col1, col2, col3 = st.columns(3)
        box_height = 320

        with col1:
            st.markdown("*Input Text*")
            st.text_area("input_box", value=input_text, height=box_height, key="input_col")

        with col2:
            st.markdown("*Generated Text (differences highlighted)*")
            highlighted = highlight_word_level(reference_text, generated_text)
            # show highlighted HTML in a scrollable div
            st.markdown(
                f"<div style='height:{box_height}px; overflow:auto; border:1px solid #ddd; padding:8px'>{highlighted}</div>",
                unsafe_allow_html=True
            )
            st.markdown("*Translated Generated Text*")
            st.write(translated_text if translated_text else "—")

        with col3:
            st.markdown("*Reference Text*")
            st.text_area("reference_box", value=reference_text, height=box_height, key="ref_col")

        # -------------------------------
        # Readability Bar comparison
        # -------------------------------
        st.subheader("Readability Scores Comparison")
        df_rd = pd.DataFrame({
            "Metric": ["Flesch", "GunningFog"],
            "Input Text": [rd_input["Flesch"], rd_input["GunningFog"]],
            "Generated Text": [rd_generated["Flesch"], rd_generated["GunningFog"]]
        }).set_index("Metric")
        fig_rd, ax_rd = plt.subplots(figsize=(7,4))
        df_rd.plot(kind='bar', ax=ax_rd)
        ax_rd.set_ylabel("Score")
        ax_rd.set_title("Readability Comparison")
        st.pyplot(fig_rd)

        # -------------------------------
        # Roughness radar chart (Cosine, Jaccard, BLEU, ROUGE-L)
        # -------------------------------
        st.subheader("Roughness Metrics Radar Chart (against Reference)")
        labels = ['Cosine', 'Jaccard', 'BLEU', 'ROUGE-L']
        stats = [cos_sim * 100, jacc_sim * 100, bleu_v * 100, rouge_l_v * 100]
        # normalize and create radar
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]
        angles += angles[:1]
        fig_r, ax_r = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        ax_r.plot(angles, stats, 'o-', linewidth=2)
        ax_r.fill(angles, stats, alpha=0.25)
        ax_r.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax_r.set_ylim(0, 100)
        st.pyplot(fig_r)

        # -------------------------------
        # Option to download side-by-side results
        # -------------------------------
        export_df = pd.DataFrame({
            "Input Text": [input_text],
            "Generated Text": [generated_text],
            "Reference Text": [reference_text],
            "Translated Generated": [translated_text],
            "Gen Summary": [gen_summary],
            "Cosine": [cos_sim],
            "Jaccard": [jacc_sim],
            "BLEU": [bleu_v],
            "ROUGE-L": [rouge_l_v]
        })
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download comparison CSV", data=csv_bytes, file_name="comparison.csv", mime="text/csv")