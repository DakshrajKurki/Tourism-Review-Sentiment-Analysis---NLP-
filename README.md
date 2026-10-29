# 🏨 Tourism Review Sentiment Analysis (NLP)

Classifying **20,491 TripAdvisor hotel reviews** as Positive, Neutral or Negative using a classic NLP pipeline, and comparing four text-classification models.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

## Overview

Hotels and travel platforms collect far more reviews than anyone can read. This project turns raw review text into sentiment labels so that overall guest satisfaction, and the words that drive it, can be measured at scale.

Ratings are mapped to sentiment (1–2 → Negative, 3 → Neutral, 4–5 → Positive), the text is cleaned through a 10-step preprocessing pipeline, and two feature representations are tested with two classifiers.

## Pipeline

1. Load and inspect the dataset, plot the rating distribution
2. Create sentiment labels from star ratings
3. Preprocess text: lower-casing, punctuation/number removal, tokenisation, stop-word removal, stemming/lemmatisation
4. Word clouds for positive vs negative reviews
5. 80/20 train–test split (16,392 / 4,099 reviews)
6. Feature extraction with **Bag of Words** and **TF-IDF** (5,000 features each)
7. Train and compare four models, plot confusion matrices
8. Predict sentiment for new, unseen reviews

## Results

| Model | Features | Accuracy | Weighted F1 |
|---|---|---|---|
| Naive Bayes | Bag of Words | 81.1% | 0.818 |
| Naive Bayes | TF-IDF | 81.7% | 0.763 |
| Logistic Regression | Bag of Words | 82.4% | 0.820 |
| **Logistic Regression** | **TF-IDF** | **85.8%** | **0.830** |

Logistic Regression with TF-IDF performed best. The Neutral class is the hardest to predict (it is the smallest class and reviews rated 3 often mix praise and complaints), which is why macro-averaged scores are lower than accuracy.

## Repository contents

| File | Description |
|---|---|
| `NLP PROJECT.ipynb` | Full notebook: preprocessing, features, models, evaluation |
| `tripadvisor_hotel_reviews.csv` | Dataset — 20,491 reviews with 1–5 star ratings |
| `NLP Report Final.pdf` | Written project report |

## Run it

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
jupyter notebook "NLP PROJECT.ipynb"
```
Update the CSV path in the "Load Dataset" cell to point to `tripadvisor_hotel_reviews.csv` in this folder.

---
Course project · Natural Language Processing (E1CAA312), BCA Semester IV, Alliance University

**Dakshraj Singh Chandawat** · [GitHub](https://github.com/DakshrajKurki) · [LinkedIn](https://www.linkedin.com/in/dakshraj-singh-chandawat-0a6349314)
