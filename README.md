# README

---

### Important

This repository is associated with the dissertation of a university student ("Fluxx in Reinforcement Learning"). This README is currently written with examiners in mind, who have access to the associated report. Eventually, the report will likely be made public - when that happens, this README will be updated with further details.

### Setup

1. Make a venv.
2. Install PyTorch for your CUDA version: https://pytorch.org/get-started/locally/ (rely on pip at your own peril)
3. Call `pip install -e .`
4. (Optional) the repo comes with embedding tables pre-generated. If you want to create your own, install the transformer model used with the following command:
```
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', truncate_dim=64); m.save('transformer_models/nomic-embed')"
```
5. You're done!

### Usage

Associated report (part II project) contains a breakdown of the repository structure.

To get started with scripts, call one of the following:
- `eval_agents.py -h`
- `eval_agents_with_puzzles.py -h`
- `train.py -h`

### Extra notes

- The experiment results were too large to store in the zip. Call `git lfs pull` to access then.
- Call `tensorboard --logdir=[DIRNAME]` to look at run results.
  - `DIRNAME` likely `experiments` or `final_experiments`.