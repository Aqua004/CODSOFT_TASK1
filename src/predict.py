import argparse
from pathlib import Path

import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/genre_model.joblib')
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    if not Path(args.model).exists():
        parser.error('Model not found; run src/train.py first')
    model = joblib.load(args.model)
    print(model.predict([args.text])[0])


if __name__ == '__main__':
    main()
