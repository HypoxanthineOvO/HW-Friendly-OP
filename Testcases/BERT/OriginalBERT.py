
"""
使用 BERT 模型(HuggingFace Transformers)对 SQuAD 验证集前 50 条样本进行问答预测，并计算 EM/F1。

"""
import argparse
import json

import torch
from datasets import load_dataset
import evaluate  # evaluate>=0.4.0
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
# from datasets import load_from_disk


# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepset/bert-base-uncased-squad2",
        help="Hugging Face model repo or local path fine-tuned on SQuAD.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0 if torch.cuda.is_available() else -1,
        help="CUDA device index; -1 代表 CPU。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation[:50]",
        help="要加载的 SQuAD 子集。",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading dataset split: {args.split} …")
    dataset = load_dataset("squad", split=args.split)
    # dataset = load_from_disk("squad_disk")["validation"]
    # dataset = dataset.select(range(50))

    print(f"Loading model & tokenizer: {args.model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_id)

    qa_pipe = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=16,
    )

    predictions = []
    references = []
    merged_rows = []

    print("Running inference …")
    for sample in dataset:
        pred = qa_pipe({"question": sample["question"], "context": sample["context"]})
        predictions.append({"id": sample["id"], "prediction_text": pred["answer"]})
        references.append({"id": sample["id"], "answers": sample["answers"]})

        merged_rows.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "prediction": pred["answer"],
                "references": sample["answers"]["text"] if isinstance(sample["answers"], dict) else sample["answers"],
            })

    metric = evaluate.load("squad")
    scores = metric.compute(predictions=predictions, references=references)

    print("\n======= Evaluation on first 50 SQuAD samples =======")
    print(f"Exact Match (EM): {scores['exact_match']:.2f}")
    print(f"F1 Score: {scores['f1']:.2f}")

    print("\n======= Detailed predictions (prediction ↔ references) =======")
    for row in merged_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
