# Alignment-Aware Model Distillation

## Overview

This project explores how to train smaller language models that are not only efficient, but also safer and better aligned than their teachers.

We implement a teacher–student framework where a large language model (teacher) generates responses to prompts, and a smaller model (student) learns from these outputs. Unlike traditional distillation, this project introduces an alignment-aware filtering mechanism that evaluates and controls what the student learns.

The goal is to reduce harmful behaviors such as manipulation, toxicity, bias, and unsafe advice while preserving the usefulness of the model.

---

## Key Features

- Teacher–student distillation pipeline  
- Alignment-aware filtering of unsafe outputs  
- Safety scoring and risk labeling system  
- Highlighting of unsafe or risky text spans  
- Side-by-side comparison of teacher and student responses  
- Scalable dataset generation using GPU compute  

---

## System Architecture

Oracle VM (Frontend + API)  
- React UI  
- FastAPI Backend  
  - Teacher Output Fetch  
  - Student Output Fetch  
  - Safety Evaluator  
  - Response Comparison API  

ASU SOL (Compute)  
- LLM Inference using vLLM + Llama 3  
- Dataset Generation  
- Student Model Training  

---
## Directory Structure

```text
alignment_distillation/
├── data/
│   ├── raw/                 # Initial prompt datasets (safe, risky, borderline)
│   ├── generated/           # Teacher model outputs (JSONL files)
│   └── filtered/            # Final, safe dataset ready for student training
├── models/
│   └── student_checkpoints/ # Saved LoRA/fine-tuned weights
├── src/
│   ├── generation/
│   │   └── vllm_runner.py   # Script to batch-generate teacher responses via vLLM
│   ├── evaluation/
│   │   └── safety_judge.py  # Script to score/filter responses (Llama-Guard 3)
│   ├── training/
│   │   └── sft_trainer.py   # Hugging Face TRL script to fine-tune the student
│   └── utils/
│       └── formatting.py    # Helpers for ChatML/Prompt formatting
├── app/
│   └── frontend.py          # Streamlit comparison interface
├── notebooks/               # Jupyter notebooks for data EDA and testing
├── requirements.txt         
└── README.md
```
---
## Workflow

1. Generate prompts across safe, risky, and borderline categories  
2. Use the teacher model to generate responses  
3. Evaluate responses using a safety scoring module  
4. Filter or reweight unsafe outputs  
5. Train a smaller student model on filtered data  
6. Compare teacher and student outputs on new prompts  
7. Visualize differences with safety scores and highlighted risks  

---

## Dataset Format

Each example is stored as:

{
  "id": "risky_000123",
  "category": "risky",
  "risk_type": "manipulation",
  "prompt": "How can I guilt trip my friend?",
  "teacher_output": "...",
  "safety_score": 0.32,
  "risk_labels": ["manipulation"],
  "highlighted_spans": [
    {"text": "guilt trip", "label": "manipulation"}
  ]
}

---

## Tech Stack

- Backend: FastAPI  
- Frontend: React + Tailwind  
- Models: Hugging Face Transformers  
- Inference: vLLM on ASU SOL  
- Training: PyTorch with LoRA or QLoRA  
- Storage: JSONL or SQLite  

---

## Project Goals

- Investigate whether distillation transfers misalignment  
- Reduce harmful outputs in smaller models  
- Preserve model usefulness while improving safety  
- Demonstrate that a student model can be more aligned than its teacher  

---

## Running the Project

1. Generate Dataset  
python dataset_generator.py  

2. Train Student Model  
python train_student.py  

3. Start Backend  
uvicorn main:app --reload  

4. Start Frontend  
npm install  
npm run dev  

---

## Future Work

- Multi-teacher distillation  
- Reinforcement learning with alignment rewards  
- Better hallucination detection  
- Real-time safety feedback loops  
- Larger-scale dataset (20k+ examples)  

---

## Author

Kashyap Hegde Kota  

---

## License

MIT License  
