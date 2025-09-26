evolve - Minimal multiplayer PD population + PPO trainer

This repository contains a small research/simulation project that implements
a population of agents playing a prisoner's dilemma-like payoff using a
transformer-based policy and a synchronous PPO trainer.

Contents
- `env.py` - population environment / simulation loop
- `agent.py` - agent wrapper and per-agent cache/history
- `models.py` - `PolicyTransformer` model (transformer-based policy + value head)
- `trainer.py` - synchronous PPO trainer (trains on sequences dumped from agents)
- `buffer.py` - global replay buffer
- `utils.py` - helper encodings and payoff table
- `requirements.txt` - minimal Python dependencies

Quickstart
1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run a short simulation (will print progress):

```powershell
python -c "from env import run_sim; run_sim(steps=2000, N=30, history_len=4, pairs_per_step=10, train_every=50, verbose=True)"
```

How to create a public GitHub repo and push this project

Option A — using the GitHub CLI (`gh`) (recommended):

```powershell
# initialize git locally (if not already a repo)
git init
git add .
git commit -m "Initial commit"
# create a public repo under your user account and push
# replace <your-username> and <repo-name>
gh repo create <your-username>/<repo-name> --public --source=. --remote=origin --push
```

Option B — using GitHub website:
1. Create a new repository on https://github.com/new (give it a name, description, set Public).
2. On your machine:
```powershell
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Notes and suggestions
- The `trainer` uses PyTorch and expects `torch` and `numpy` to be installed (see `requirements.txt`).
- If you want me to create CI (GitHub Actions) to run lints/tests on push, I can add workflow files.
- If you want, I can also prepare a small `setup.py` / `pyproject.toml` for packaging or add unit tests.

Contact
If you'd like me to perform the push for you, I can't create remote GitHub repositories from here — you'll need to run the `gh` or `git` commands above from your machine (authenticated). I can help with the exact commands and files to add or automate client-side steps if you want.
