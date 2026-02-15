
import pandas as pd
import nltk
import os

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

current_dir = os.path.dirname(os.path.abspath(__file__))

file1_path = os.path.join(current_dir, "Cogvinia_response.xlsx")
file1_column = "assistant1"
# Replace with your path to test dataset
file2_path = os.path.join(current_dir, "test.xlsx")
file2_column = "assistant1"

output_path = os.path.join(current_dir,  "evaluation_with_NLP.xlsx")

print("Reading files...")
print(f"Looking for: {file1_path}")
print(f"Looking for: {file2_path}")

df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)

preds = df1[file1_column].dropna().astype(str).tolist()
refs = df2[file2_column].dropna().astype(str).tolist()

print(f"First file has {len(preds)} predictions")
print(f"Second file has {len(refs)} references")

min_len = min(len(preds), len(refs))
preds = preds[:min_len]
refs = refs[:min_len]
print(f"Evaluating {min_len} pairs")

print("\nFirst 3 comparison examples:")
for i in range(min(3, min_len)):
    print(f"Example {i + 1}:")
    print(f"  Model output: {preds[i]}")
    print(f"  Reference: {refs[i]}")
    print()

from rouge_score import rouge_scorer
import sacrebleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(scorer.score(ref, pred))

    avg_scores = {}
    for key in ['rouge1', 'rouge2', 'rougeL']:
        avg_scores[key] = sum([s[key].fmeasure for s in scores]) / len(scores)
    return avg_scores

def compute_bleu_selected(predictions, references):
    refs = [[ref] for ref in references]
    results = {}

    bleu1 = sacrebleu.metrics.BLEU(max_ngram_order=1)
    bleu1_score = bleu1.corpus_score(predictions, refs)
    results["bleu1"] = bleu1_score.score / 100.0

    bleu2 = sacrebleu.metrics.BLEU(max_ngram_order=2)
    bleu2_score = bleu2.corpus_score(predictions, refs)
    results["bleu2"] = bleu2_score.score / 100.0

    return results

def compute_glue_score(predictions, references):
    try:
        vectorizer = TfidfVectorizer(max_features=1000).fit(predictions + references)
        pred_vecs = vectorizer.transform(predictions)
        ref_vecs = vectorizer.transform(references)

        semantic_scores = []
        for i in range(len(predictions)):
            score = cosine_similarity(pred_vecs[i], ref_vecs[i])[0][0]
            semantic_scores.append(score)

        semantic_similarity = np.mean(semantic_scores)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        refs = [[ref] for ref in references]
        bleu = sacrebleu.corpus_bleu(predictions, refs)

        weights = {
            'semantic': 0.3,
            'rouge1': 0.15,
            'rouge2': 0.15,
            'rougeL': 0.15,
            'bleu': 0.25
        }

        glue_score = (
            weights['semantic'] * semantic_similarity +
            weights['rouge1'] * np.mean(rouge1_scores) +
            weights['rouge2'] * np.mean(rouge2_scores) +
            weights['rougeL'] * np.mean(rougeL_scores) +
            weights['bleu'] * (bleu.score / 100.0)
        )

        return {"GLUE": glue_score}

    except Exception as e:
        print(f"GLUE score calculation failed: {e}")
        return {"GLUE": 0}

print("Calculating evaluation metrics...")
rouge_results = compute_rouge(preds, refs)
bleu_results = compute_bleu_selected(preds, refs)
glue_results = compute_glue_score(preds, refs)

all_results = {
    **rouge_results,
    **bleu_results,
    **glue_results,
    "sample_count": min_len
}

print("\n" + "=" * 50)
print("Evaluation Results Summary")
print("=" * 50)

print(f"\n📊 File Information:")
print(f"  Prediction file: {os.path.basename(file1_path)} (column: '{file1_column}')")
print(f"  Reference file: {os.path.basename(file2_path)} (column: '{file2_column}')")
print(f"  Number of samples: {min_len}")

print(f"\n🎯 Text Quality Metrics:")
print(f"  ROUGE-1:  {all_results.get('rouge1', 0):.4f}")
print(f"  ROUGE-2:  {all_results.get('rouge2', 0):.4f}")
print(f"  ROUGE-L:  {all_results.get('rougeL', 0):.4f}")
print(f"  BLEU-1:   {all_results.get('bleu1', 0):.4f}")
print(f"  BLEU-2:   {all_results.get('bleu2', 0):.4f}")
print(f"  GLUE:     {all_results.get('GLUE', 0):.4f}")

results_df = pd.DataFrame([all_results])
results_df.to_excel(output_path, index=False)
print(f"\n✅ Results saved to: {os.path.basename(output_path)}")
print(f"Full path: {output_path}")
