# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **FIU Sprinternship Advanced Data Science & AI Workshop** — an educational curriculum built around Jupyter notebooks. It progresses through data loading, preparation, modeling, and file I/O (Lessons 1–5), ending with six capstone industry-simulation challenges in `Lesson5-Challenges/`.

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Running Notebooks

Open and run notebooks with Jupyter:
```bash
jupyter notebook
```

All lesson content and challenge work is done inside `.ipynb` files — there are no standalone Python scripts.

## Lesson Structure

- **lesson_1.ipynb** — Data loading: CSV, APIs, web scraping, Kaggle datasets
- **lesson_2.ipynb** — Data preparation and baseline modeling
- **lesson_3.ipynb / lesson_4.ipynb** — Advanced modeling and evaluation
- **lesson_5.ipynb** — File formats: JSON, YAML, Markdown I/O

## Challenges (`Lesson5-Challenges/`)

Six capstone projects following the CRISP-DM lifecycle (Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Recommendations):

| # | Folder | Domain |
|---|--------|--------|
| 1 | `1_ClinicalPriority/` | Healthcare — heart disease classification |
| 2 | `2_TransactionRisk/` | Finance — transaction risk assessment |
| 3 | `3_ProductRecommendationSystem/` | E-commerce — recommendations |
| 4 | `4_DeliveryDelayPrediction/` | Logistics — delay prediction |
| 5 | `5_ContentPerformancePrediction/` | Media — content analytics |
| 6 | `6_StudentRetentionDropout/` | Education — retention prediction |

Each challenge folder contains:
- `Challenge[N]_[Name].md` — Problem statement and business context
- `notebook.ipynb` — Template notebook for the team solution

## Tech Stack

- **Data:** pandas, numpy, scipy
- **ML:** scikit-learn, xgboost
- **Visualization:** matplotlib, seaborn
- **Data acquisition:** kaggle API, requests, beautifulsoup4
- **Serialization:** PyYAML, joblib

## Kaggle Datasets

Challenges use Kaggle datasets. To download:
```bash
kaggle datasets download -d {dataset_slug}
# Example:
kaggle datasets download -d johnsmith88/heart-disease-dataset
```

Requires `~/.kaggle/kaggle.json` credentials (not committed to the repo).
