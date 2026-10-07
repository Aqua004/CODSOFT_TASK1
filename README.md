# Movie Genre Classification (CodSoft Task 1)

Classify movie genres from plot descriptions using the IMDb genre dataset: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

## Dataset
Download the dataset and place `train_data.txt` in `data/`. Expected line format: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`. The test file lacks labels; this project uses an 80/20 stratified holdout split from the training file for evaluation. Do not commit raw data.

## Methodology
- Text features: TF-IDF with unigram and bigram features.
- Model: class-weighted Logistic Regression.
- Evaluation: stratified 80/20 train/test split with `random_state=42`.
- Metrics: accuracy, macro F1, weighted F1, and per-genre precision/recall/F1.

## Run
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --data data/train_data.txt
```

## Verified Results
The project was run locally on the downloaded dataset.

- Test samples: 10,821
- Accuracy: 0.53
- Macro F1-score: 0.3882
- Weighted F1-score: 0.54

Higher per-class F1 scores included Western (0.79), Game-show (0.75), Documentary (0.74), and Horror (0.62). Lower scores occurred for several rare or semantically overlapping genres, including Biography (0.04), Fantasy (0.13), Mystery (0.13), and Musical (0.17). Macro F1 is the most representative aggregate score here because the genre classes are imbalanced.

## Output
Training saves `models/genre_model.joblib` and `results/metrics.json` locally. Both are excluded from Git.
