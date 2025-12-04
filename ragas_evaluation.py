from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextRelevance,
    ContextRecall
)
import json

# --------------------------
# Load evaluation data
# --------------------------
with open("ragas_eval_sample.json", "r") as f:
    data = json.load(f)

eval_rows = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": []
}

for item in data:
    eval_rows["question"].append(item["question"])
    eval_rows["answer"].append(item["answer"])
    eval_rows["contexts"].append(item["contexts"])
    eval_rows["ground_truth"].append(item["ground_truth"])

dataset = Dataset.from_dict(eval_rows)

# --------------------------
# Run RAGAS evaluation
# --------------------------
metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    ContextRelevance(),
    ContextRecall(),
]

result = evaluate(dataset, metrics=metrics)

print("\n===== RAGAS Evaluation Results =====")
print(result)

try:
    print("\n=== Detailed Pandas Table ===")
    print(result.to_pandas())
except Exception:
    pass
