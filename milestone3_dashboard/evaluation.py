from rouge_score import rouge_scorer
import sacrebleu
import textstat
import plotly.graph_objects as go

def evaluate_outputs(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    results = {"ROUGE-1": 0, "ROUGE-L": 0, "BLEU": 0}

    # ROUGE
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        results["ROUGE-1"] += score["rouge1"].fmeasure
        results["ROUGE-L"] += score["rougeL"].fmeasure

    n = len(predictions)
    results["ROUGE-1"] /= n
    results["ROUGE-L"] /= n

    # BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    results["BLEU"] = bleu.score

    return results

def plot_radar(scores):
    fig = go.Figure()
    metrics = list(scores.keys())
    values = list(scores.values())
    fig.add_trace(go.Scatterpolar(r=values, theta=metrics, fill="toself", name="Scores"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    return fig

def plot_readability(predictions):
    metrics = {
        "Flesch": sum(textstat.flesch_reading_ease(t) for t in predictions) / len(predictions),
        "Gunning Fog": sum(textstat.gunning_fog(t) for t in predictions) / len(predictions),
        "SMOG": sum(textstat.smog_index(t) for t in predictions) / len(predictions),
    }
    fig = go.Figure([go.Bar(x=list(metrics.keys()), y=list(metrics.values()))])
    fig.update_layout(title="Readability Scores", xaxis_title="Metric", yaxis_title="Score")
    return fig
