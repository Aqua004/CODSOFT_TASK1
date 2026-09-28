# Movie Genre Classification (CodSoft Task 1)

Classify a movie's primary genre from its plot description using TF-IDF and logistic regression. This is a reproducible baseline, not a claim of a measured score.

## Dataset
Download the CodSoft Task 1 movie genre dataset from the link in the internship PDF. Place `train_data.txt` in `data/` (not committed). Each line is expected to be `ID ::: TITLE ::: GENRE ::: DESCRIPTION`. The supplied test file lacks labels; use held-out training rows for honest evaluation.

## Run
```bash
python -m venv .venv
# Activate .venv for your operating system
pip install -r requirements.txt
python src/train.py --data data/train_data.txt
python src/predict.py --model models/genre_model.joblib --text 'A detective investigates a mysterious disappearance.'
```

The training script saves `models/genre_model.joblib` and `results/metrics.json`. It prints accuracy, macro F1, weighted F1, and a classification report. Review per-class recall before claiming performance. Split before vectorization; TF-IDF is fitted only on training examples.

## Demo / submission
Show the dataset format, a fresh training run, evaluation, and a prediction in your video. Add your measured scores here only after running the code. Do not commit the dataset or trained model if licensing or size prevents it.

## Results
Not yet run or independently verified.
