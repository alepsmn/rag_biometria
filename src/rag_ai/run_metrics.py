import json
from pathlib import Path

BASEDIR = Path(__file__).resolve().parent
results_dir = BASEDIR.parent.parent / "eval" / "retrieval_results.json"

with open(results_dir, 'r', encoding='utf-8') as retrieval_results:
    results = json.load(retrieval_results)


metrics_answerable = {}
metrics_non_answerable = {}

for result in results:
    question_id = result["question_id"]
    evidence_pages = set(result["evidence_pages"])
    retrieved = result["retrieved_contexts"]

    relevance = [
        item["metadata"] in evidence_pages
        for item in retrieved
    ]

    metrics = {}

    for k in [2, 4, 6]:
        top_k = relevance[:k]
        metrics[f"hit_rate@{k}"] = int(any(top_k))

    rr = 0
    for position, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            rr = 1 / position
            break
    metrics["RR"] = rr

    scores = [item["score"] for item in retrieved]
    metrics["max_score"] = (
        max(scores) if scores else None
    )

    if result["answerable"]:
        metrics_answerable[question_id] = metrics
    else:
        metrics_non_answerable[question_id] = metrics

# mover los datos fuera de la estructura total
hr2 = 0
hr4 = 0
hr6 = 0
max_score = 0
min_score = 1
mrr = 0
n_questions = len(metrics_answerable) # antes de mutar las claves
for key in metrics_answerable:
    hr2 += metrics_answerable[key]["hit_rate@2"]
    hr4 += metrics_answerable[key]["hit_rate@4"]
    hr6 += metrics_answerable[key]["hit_rate@6"]
    mrr += metrics_answerable[key]["RR"]
    if metrics_answerable[key]["max_score"] > max_score:
        max_score = metrics_answerable[key]["max_score"]
    if metrics_answerable[key]["max_score"] < min_score:
        min_score = metrics_answerable[key]["max_score"]
metrics_answerable["group_metrics"] = {}
metrics_answerable["group_metrics"]["hr2"] = hr2 / n_questions
metrics_answerable["group_metrics"]["hr4"] = hr4 / n_questions
metrics_answerable["group_metrics"]["hr6"] = hr6 / n_questions
metrics_answerable["group_metrics"]["hr6"] = hr6 / n_questions

metrics_answerable["group_metrics"]["MRR"] = mrr / n_questions
metrics_answerable["group_metrics"]["max_answerable_score"] = max_score
metrics_answerable["group_metrics"]["min_answerable_score"] = min_score

max_score = 0
for key in metrics_non_answerable:
    if metrics_non_answerable[key]["max_score"] > max_score:
        max_score = metrics_non_answerable[key]["max_score"]
metrics_non_answerable["group_metrics"] = {}
metrics_non_answerable["group_metrics"]["max_non_answerable_score"] = max_score

output_dir = BASEDIR.parent.parent / "eval" / "retrieval_metrics.json"
with open(output_dir, "w", encoding="utf-8") as file:
    json.dump([metrics_answerable, metrics_non_answerable], file, ensure_ascii=False, indent=4)