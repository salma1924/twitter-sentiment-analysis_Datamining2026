# twitter-sentiment-analysis_Datamining2026

# 🐦 Twitter Sentiment Analysis — Data Mining Project

> **University Data Mining Project 
> Sentiment classification, unsupervised clustering, and explainable AI on 1.6 million tweets.

---

## 📌 Project Overview

This project applies data mining techniques to the **Sentiment140** dataset — a large-scale collection of 1.6 million tweets labeled as positive or negative. The goal is to build a robust, interpretable sentiment analysis pipeline that answers real-world questions about public opinion expressed on social media.

We combine **supervised classification**, **unsupervised clustering**, and **exploratory text mining** to extract meaningful insights from raw tweet data. Model predictions are explained using SHAP and/or LIME to ensure transparency and interpretability.

---

## 📂 Dataset Description

| Property | Value |
|---|---|
| **Source** | [Sentiment140 — Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) |
| **Size** | 1,600,000 tweets |
| **Labels** | 0 = Negative, 4 = Positive (binary) |
| **Features** | target, id, date, flag, user, text |
| **Language** | English |
| **Format** | CSV |


---

## 🔬 Mining Techniques

### Classification (Supervised)

**Best model selection** is based on F1-score, accuracy, precision, and recall. The winning model is then explained using **SHAP** and/or **LIME**.

### Clustering (Unsupervised)
- Labels are **ignored** to discover hidden tweet groups

### Text Mining & EDA
- Class distribution
- Tweet length distributions (words & characters)
- Most frequent words (pre/post stopword removal)
- Word clouds per sentiment class
- Top hashtags and their sentiment profiles
- Correlation heatmap (length, hashtag count, punctuation, sentiment)
- Outlier detection (extremely short/long tweets)

---

## 🗂️ CRISP-DM Structure

```
Phase 1 — Business Understanding    ← Current Phase
Phase 2 — Data Understanding (EDA)
Phase 3 — Data Preparation
Phase 4 — Modeling
Phase 5 — Evaluation
Phase 6 — Deployment (Dashboard)
```

---


---

## ⚙️ Environment Setup

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

1. Visit [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. Download `training.1600000.processed.noemoticon.csv`
3. Place it in `data/raw/`

### Run the Notebooks

```bash
jupyter notebook notebooks/
```

### Run the Dashboard (Phase 6)

```bash
streamlit run dashboard/app.py
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
wordcloud>=1.9.0
nltk>=3.8.0
shap>=0.41.0
lime>=0.2.0
streamlit>=1.20.0
jupyter>=1.0.0
plotly>=5.13.0
```

---

## 👥 Team

Student ID1: 230102006
Student ID2: 230101802

---

## 📄 License

This project is developed for academic purposes only.
