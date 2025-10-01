from transformers import pipeline

def generate_text(text, model_choice, task, lang=None):
    if task == "Summarization":
        summarizer = pipeline("summarization", model=model_choice)
        return summarizer(text, max_length=100, min_length=20, do_sample=False)[0]["summary_text"]

    elif task == "Paraphrasing":
        paraphraser = pipeline("text2text-generation", model=model_choice)
        return paraphraser(f"paraphrase: {text}", max_length=100, do_sample=False)[0]["generated_text"]

    elif task == "Translation":
        translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{lang}")
        return translator(text)[0]["translation_text"]

    return text
