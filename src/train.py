import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_data(path):
    rows = []
    with open(path, encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            parts = line.rstrip('\n').split(' ::: ', 3)
            if len(parts) != 4:
                raise ValueError(f'Invalid row {line_number}: expected ID ::: TITLE ::: GENRE ::: DESCRIPTION')
            rows.append((parts[2].strip(), parts[3].strip()))
    frame = pd.DataFrame(rows, columns=['genre', 'description'])
    frame = frame.replace('', pd.NA).dropna().drop_duplicates()
    if frame.empty or frame['genre'].nunique() < 2:
        raise ValueError('Need non-empty descriptions from at least two genres')
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train_data.txt')
    args = parser.parse_args()
    frame = load_data(args.data)
    counts = frame.genre.value_counts()
    if counts.min() < 2:
        raise ValueError('Each genre needs at least two rows for a stratified split')
    x_train, x_test, y_train, y_test = train_test_split(
        frame.description, frame.genre, test_size=0.2, random_state=42, stratify=frame.genre
    )
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        'accuracy': accuracy_score(y_test, predicted),
        'macro_f1': f1_score(y_test, predicted, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test, predicted, average='weighted', zero_division=0),
        'classification_report': classification_report(y_test, predicted, output_dict=True, zero_division=0),
    }
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    joblib.dump(model, 'models/genre_model.joblib')
    Path('results/metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(classification_report(y_test, predicted, zero_division=0))
    print(f'Macro F1: {metrics["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
