---
title: AIE10 Ensemble Methods
emoji: 🌳
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: Home.py
pinned: false
---

# AIE10 — Ensemble Methods Visualizations

Animated walkthroughs of core ensemble learning algorithms for the **Applied Artificial Intelligence** course.

## Topics
1. **Decision Trees** — recursive splitting, information gain, tree growth
2. **Bagging** — bootstrap sampling, parallel learners, variance reduction
3. **Boosting** — sequential weak learners, sample reweighting (AdaBoost)
4. **Stacking** — base learners + meta-learner
5. **XGBoost** — gradient boosting on trees, regularization

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```

## Structure
- `Home.py` — landing page
- `pages/` — one walkthrough per algorithm
- `utils/` — shared helpers (data, plotting, animation)
