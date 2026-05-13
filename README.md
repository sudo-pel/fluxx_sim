# README

---

### Setup

1. Install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/ (rely on pip at your own peril)
2. Call `pip install -e .`
3. Install the transformer model used with the following command:
```
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', truncate_dim=64); m.save('transformer_models/nomic-embed')"
```
4. You're done!

### Usage

Associated report (part II project) contains a breakdown of the repository structure.

To get started with scripts, call one of the following:
- `eval_agents.py -h`
- `eval_agents_with_puzzles.py -h`
- `train.py -h`