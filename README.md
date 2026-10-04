# Arabic Tweet Sentiment Analysis: Naive Bayes Baseline

![Python](https://img.shields.io/badge/python-3.x-blue) ![NLTK](https://img.shields.io/badge/NLTK-NLP-green) ![Arabic](https://img.shields.io/badge/language-Arabic-lightgrey)

Binary sentiment classification (positive/negative) of **Arabic tweets** using a bag-of-words model and NLTK's Naive Bayes classifier. It is a simple, fast and interpretable baseline for Arabic NLP, which still has far fewer open resources than English.

## Results (held-out test set)

| Metric | Value |
|---|---|
| Accuracy | **89.1%** |
| Positive precision / recall | 0.920 / 0.861 |
| Negative precision / recall | 0.865 / 0.923 |
| Positive F1 | 0.890 |

## Approach

1. Load the labelled train and test TSVs (`input/`, positive and negative tweets).
2. Normalise the text: strip diacritics, URLs, mentions, non-Arabic characters and repeated letters.
3. Extract unigram (configurable n-gram) bag-of-words features.
4. Train `nltk.NaiveBayesClassifier` and report precision, recall and F-score per class.
5. Inspect the most informative features to understand what drives sentiment.

## Run it

```bash
pip install nltk pandas numpy
jupyter notebook arabic-sentiment-analysis-in-tweets-nb-bow.ipynb
```

## Ideas for next steps

- Compare against TF-IDF + logistic regression, and a fine-tuned Arabic transformer such as AraBERT or CAMeLBERT.
- Handle dialectal Arabic (Egyptian, Gulf, Levantine) explicitly.
- Serve the model behind an API and track drift on new tweets.

Built while contributing to [Omdena](https://omdena.com/) Arabic NLP community projects.

---
Author: [Ayman Metwally](https://github.com/ayman-metwally2020)
