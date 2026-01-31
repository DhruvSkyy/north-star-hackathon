# FACTTRACE – AI Truth Jury

Evaluates whether a system claim is **FAITHFUL** or **MUTATED** relative to ground truth using multiple AI jury agents and a deterministic (non-LLM) arbiter.

---

## Requirements
- Python 3.10+
- Conda (Miniconda or Anaconda)
- OpenAI API key

---

## Setup (Conda)
```bash
conda create -n facttrace python=3.11 -y
conda activate facttrace
pip install -r requirements.txt
