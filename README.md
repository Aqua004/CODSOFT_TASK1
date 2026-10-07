# Movie Genre Classification (CodSoft Task 1)

Classify movie genres from plot descriptions using the IMDb genre dataset: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

## Dataset
Download the dataset and place `train_data.txt` in `data/`. Expected line format: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`. The test file lacks labels; use a holdout split from the training file for honest evaluation. Do not commit raw data.

## Run
```bash
python -m venv .venv
# Activate the virtual environment
pip install -r requirements.txt
python src/train.py --data data/train_data.txt
```
Outputs: `models/genre_model.joblib` and `results/metrics.json` (both gitignored). Reports accuracy, macro F1, weighted F1, and a per-class classification report. TF-IDF is fitted only on training examples after the split.

## Submission
Run on your downloaded dataset, record a demo video, and add measured results here only after verifying them.

## Results
Not yet run or independently verified.
