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

## Workflow

1. Generate prompts across safe, risky, and borderline categories  
2. Use the teacher model to generate responses  
3. Evaluate responses using a safety scoring module  
4. Filter or reweight unsafe outputs  
5. Train a smaller student model on filtered data  
6. Compare teacher and student outputs on new prompts  
7. Visualize differences with safety scores and highlighted risks  

---

## Implementation Steps

1. Teacher Model Setup  
   - Host Llama 3 on ASU SOL using GPU  
   - Serve using vLLM for fast inference  

2. Dataset Generation  
   - Generate prompts using templates  
   - Run teacher inference to collect responses  
   - Store results in JSONL format  

3. Safety Evaluation  
   - Assign safety scores to each response  
   - Label risks such as manipulation, toxicity, bias, unsafe advice  
   - Identify and highlight unsafe spans  

4. Alignment-Aware Filtering  
   - Remove or downweight unsafe responses  
   - Keep high-quality aligned outputs for training  

5. Student Model Training  
   - Fine-tune a smaller model such as TinyLlama or Phi  
   - Use filtered dataset for distillation  

6. Evaluation  
   - Compare teacher vs student on:  
     - Helpfulness  
     - Safety  
     - Risk frequency  
     - Alignment quality  

7. Frontend Interface  
   - Input prompt  
   - Display:  
     - Teacher output  
     - Student output  
     - Safety scores  
     - Risk labels  
     - Highlighted unsafe text  

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
