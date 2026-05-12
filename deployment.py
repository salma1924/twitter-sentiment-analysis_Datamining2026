%%writefile app.py
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import re
import string
import pandas as pd # Import pandas here for df_model usage

# NLTK imports and downloads (handle gracefully for Streamlit deployment)
import nltk
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse as sp

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Utility functions (same as before) ---
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+', '', text)                  # remove @mentions
    text = re.sub(r'#(\w+)', r'\\1', text)             # keep hashtag word
    text = re.sub(r'[^a-z\s]', '', text)              # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)

def extract_hashtags(text):
    return re.findall(r'#(\w+)', str(text).lower())

st.set_page_config(page_title="Twitter Sentiment Analysis Dashboard",
                   layout="wide")

# ---------------------------------------
# Load Model + Vectorizer + Selector + Scaler
# ---------------------------------------
# Define the local path where files are copied
LOCAL_APP_DATA_PATH = './app_data'
LOCAL_MODELS_PATH = os.path.join(LOCAL_APP_DATA_PATH, 'models')
LOCAL_PLOTS_PATH = os.path.join(LOCAL_APP_DATA_PATH, 'plots')

MODEL_PATH = os.path.join(LOCAL_MODELS_PATH, "logistic_regression__c_1_.joblib")
VECTORIZER_PATH = os.path.join(LOCAL_MODELS_PATH, "tfidf_vectoriser.joblib")
SELECTOR_PATH = os.path.join(LOCAL_MODELS_PATH, "chi2_selector.joblib")
SCALER_PATH = os.path.join(LOCAL_MODELS_PATH, "scaler.joblib")
DF_MODEL_PATH = os.path.join(LOCAL_APP_DATA_PATH, "sentiment140_cleaned.csv") # Path to cleaned data

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    selector = joblib.load(SELECTOR_PATH)
    scaler = joblib.load(SCALER_PATH)
    df_model = pd.read_csv(DF_MODEL_PATH) # Load df_model here
    # Ensure target is 0/1 (safe to run even if already remapped)
    df_model['target'] = df_model['target'].map({0: 0, 4: 1}).fillna(df_model['target']).astype(int)


    # Load model evaluation metrics (hardcoded from previous output for simplicity)
    best_model_val_f1 = 0.7734
    best_model_test_f1 = 0.7698 # Using Logistic Regression (C=1) test F1 from output
    best_model_roc_auc = 0.8503 # Using Logistic Regression (C=1) ROC-AUC from output

except FileNotFoundError:
    st.error("Error: One or more required files (model, vectorizer, scaler, or data) not found. Please ensure 'models/' directory and 'sentiment140_cleaned.csv' exist.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during loading: {e}")
    st.stop()

# ---------------------------------------
# Header
# ---------------------------------------
st.title("🐦 Interactive Twitter Sentiment Analysis Dashboard")
st.markdown("""
This dashboard provides a comprehensive view of sentiment analysis on the **Sentiment140 dataset**,
showcasing **data insights, model performance, interpretability,** and offering **real-time tweet classification.**
""")

st.divider()

# Initialize session state for global filter
if 'global_sentiment_filter' not in st.session_state:
    st.session_state['global_sentiment_filter'] = 'All'

# ---------------------------------------
# SIDEBAR NAVIGATION & GLOBAL FILTER
# ---------------------------------------
st.sidebar.header("🚀 Navigate Project Stages")
section = st.sidebar.radio("",
                           ["1. Project Overview",
                            "2. EDA & Data Insights",
                            "3. Model Performance",
                            "4. Interpretability (LIME & SHAP)",
                            "5. Real-Time Sentiment Predictor",
                            "6. Business Insights"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model in Use:** \
    <span style=\"color: #1a73e8; font-weight: bold;\">Logistic Regression (C=1)</span>",
    unsafe_allow_html=True)
st.sidebar.markdown("**Feature Engineering:** TF-IDF + 7 Engineered Features")

st.sidebar.markdown("---")
st.sidebar.subheader("Global Data Filter")
st.session_state['global_sentiment_filter'] = st.sidebar.radio(
    "Filter by Sentiment:",
    ('All', 'Positive', 'Negative'),
    key='sentiment_filter'
)
st.sidebar.markdown(f"**Current Filter:** {st.session_state['global_sentiment_filter']}")


# ---------------------------------------
# 1. Project Overview
# ---------------------------------------
if section == "1. Project Overview":
    st.header("✨ Project Overview: From Raw Tweets to Sentiment Prediction")
    st.markdown("""
    This project aims to classify the sentiment of tweets as either **positive (1)** or **negative (0)**.
    We started with a large dataset of 1.6 million tweets and built a robust machine learning pipeline.

    Here's a breakdown of the key steps:

    *   **Data Loading & Preprocessing**: Downloaded the Sentiment140 dataset, remapped target labels, and handled duplicates.
    *   **Text Cleaning**: Applied techniques like lowercasing, URL/mention removal, lemmatization, and stop-word filtering to prepare tweet text.
    *   **Exploratory Data Analysis (EDA)**: Visualized class distribution, tweet lengths, top words, and hashtag sentiment to understand the data.
    *   **Feature Engineering**: Created numerical features from raw text using TF-IDF and engineered additional features like character/word counts, hashtag counts, etc.
    *   **Feature Selection & Scaling**: Used Chi-squared for TF-IDF feature selection and StandardScaler for numerical features.
    *   **Model Training**: Trained and compared various classification models (Logistic Regression, SVM, Random Forest, etc.).
    *   **Model Evaluation**: Assessed model performance using metrics like F1-score, Confusion Matrix, and ROC curves.
    *   **Model Interpretability**: Employed SHAP and LIME to understand global feature importances and explain individual predictions.
    *   **Clustering**: Explored tweet clusters in a reduced dimension space to uncover natural groupings.
    *   **Deployment**: Built this interactive Streamlit dashboard to demonstrate the solution!
    """)


# ---------------------------------------
# 2. EDA & Data Insights (Mining Technique 1)
# ---------------------------------------
elif section == "2. EDA & Data Insights":
    st.header("📊 Data Mining Technique 1: Exploratory Data Analysis & Feature Engineering")
    st.markdown("""
    Understanding the dataset is crucial. These visualizations reveal key characteristics of the tweets and their sentiments,
    as well as insights gained from engineered features.
    """)

    # Apply global filter for dynamic plots
    filtered_df = df_model.copy()
    if st.session_state['global_sentiment_filter'] == 'Positive':
        filtered_df = filtered_df[filtered_df['target'] == 1]
    elif st.session_state['global_sentiment_filter'] == 'Negative':
        filtered_df = filtered_df[filtered_df['target'] == 0]

    st.subheader("Interactive Sentiment Class Distribution")
    st.info(f"This interactive chart shows the distribution of tweets by sentiment, affected by the '{st.session_state['global_sentiment_filter']}' filter.")

    if not filtered_df.empty:
        # Remap target for display
        display_df = filtered_df['target'].map({0: 'Negative', 1: 'Positive'}).value_counts().reset_index()
        display_df.columns = ['Sentiment', 'Count']
        st.bar_chart(display_df.set_index('Sentiment'))
    else:
        st.warning("No data to display for the current filter selection.")


    st.subheader("Tweet Length Distributions")
    st.info("We analyzed character and word counts to see if tweet length correlates with sentiment.")
    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eda_02_length_distributions.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eda_02_length_distributions.png"), caption="Distribution of Tweet Lengths by Sentiment")
    else:
        st.warning("Tweet length distribution plot not found.")

    st.subheader("Top Words by Sentiment")
    st.info("These bar charts highlight the most frequent words in negative vs. positive tweets after cleaning.")
    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eda_03_top_words.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eda_03_top_words.png"), caption="Top 30 Words in Negative and Positive Tweets")
    else:
        st.warning("Top words plot not found.")

    st.subheader("Word Clouds")
    st.info("Visual representation of word frequency, where larger words appear more often in that sentiment category.")
    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eda_04_wordclouds.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eda_04_wordclouds.png"), caption="Word Clouds for Negative and Positive Tweets")
    else:
        st.warning("Word cloud image not found.")

    st.subheader("Hashtag Sentiment Profile")
    st.info("Examining the sentiment associated with popular hashtags.")
    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eda_05_hashtag_sentiment.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eda_05_hashtag_sentiment.png"), caption="Top 20 Hashtags by Sentiment Profile")
    else:
        st.warning("Hashtag sentiment plot not found.")

    st.subheader("Feature Correlation Heatmap")
    st.info("A heatmap showing the relationships between various engineered features.")
    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eda_06_correlation_heatmap.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eda_06_correlation_heatmap.png"), caption="Feature Correlation Heatmap")
    else:
        st.warning("Correlation heatmap not found.")


# ---------------------------------------
# 3. Model Performance (Mining Technique 2)
# ---------------------------------------
elif section == "3. Model Performance":
    st.header("📈 Data Mining Technique 2: Model Training & Evaluation")
    st.markdown("""
    After training various models, we rigorously evaluated their performance on unseen test data.
    Our best performing model was **Logistic Regression (C=1)**.
    """)

    st.subheader("Key Metrics for Best Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Validation F1 Score", value=f"{best_model_val_f1:.4f}")
    with col2:
        st.metric(label="Test F1 Score", value=f"{best_model_test_f1:.4f}")
    with col3:
        st.metric(label="Test ROC-AUC Score", value=f"{best_model_roc_auc:.4f}")

    st.subheader("Model Comparison")
    st.info("Comparison of F1 scores across different models on validation and test sets.")
    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_01_model_comparison.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_01_model_comparison.png"), caption="Model Comparison — Validation vs Test F1")
    else:
        st.warning("Model comparison plot not found.")

    st.subheader("Confusion Matrix")
    st.info("Visualizing the correct and incorrect predictions made by the best model.")
    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_02_confusion_matrix.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_02_confusion_matrix.png"), caption="Confusion Matrix — Logistic Regression (C=1)")
    else:
        st.warning("Confusion matrix not found.")

    st.subheader("ROC Curves")
    st.info("Receiver Operating Characteristic curves for all models, showing their discriminative power.")
    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_03_roc_curves.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_03_roc_curves.png"), caption="ROC Curves — All Models")
    else:
        st.warning("ROC curves plot not found.")


# ---------------------------------------
# 4. Interpretability (LIME & SHAP) (Mining Technique 3)
# ---------------------------------------
elif section == "4. Interpretability (LIME & SHAP)":
    st.header("💡 Data Mining Technique 3: Model Interpretability (LIME & SHAP)")
    st.markdown("""
    Understanding model predictions is as important as the predictions themselves. We used two powerful interpretability techniques:

    *   **SHAP (SHapley Additive exPlanations)**: Provides global insights into feature importance and how features influence predictions.
    *   **LIME (Local Interpretable Model-agnostic Explanations)**: Explains individual predictions by perturbing the input and observing changes.
    """)

    st.subheader("SHAP: Global Feature Importances")
    st.info("These plots show which features generally contribute most to positive or negative sentiment predictions across the dataset.")

    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_04_shap_global.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_04_shap_global.png"), caption="SHAP — Top 20 Global Feature Importances")
    else:
        st.warning("SHAP global importance plot not found.")

    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_05_shap_beeswarm.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_05_shap_beeswarm.png"), caption="SHAP Beeswarm — Feature Impact Direction")
    else:
        st.warning("SHAP beeswarm plot not found.")

    if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_06_shap_directional.png")):
        st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_06_shap_directional.png"), caption="SHAP — Words Driving Positive vs Negative Predictions")
    else:
        st.warning("SHAP directional words plot not found.")


    st.subheader("LIME: Local Prediction Explanations")
    st.info("LIME helps us understand why a specific tweet was classified as positive or negative by highlighting the most influential words in that particular tweet.")
    st.markdown("Here are examples of LIME explanations for different prediction scenarios:")

    col1, col2, col3 = st.columns(3)
    with col1:
        if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_07_lime_correct_positive.png")):
            st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_07_lime_correct_positive.png"), caption="LIME Explanation: Correctly Classified Positive Tweet")
        else:
            st.warning("LIME correct positive plot not found.")
    with col2:
        if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_07_lime_correct_negative.png")):
            st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_07_lime_correct_negative.png"), caption="LIME Explanation: Correctly Classified Negative Tweet")
        else:
            st.warning("LIME correct negative plot not found.")
    with col3:
        if os.path.exists(os.path.join(LOCAL_PLOTS_PATH, "eval_07_lime_misclassified.png")):
            st.image(os.path.join(LOCAL_PLOTS_PATH, "eval_07_lime_misclassified.png"), caption="LIME Explanation: Misclassified Tweet")
        else:
            st.warning("LIME misclassified plot not found.")


# ---------------------------------------
# 5. Real-Time Sentiment Predictor
# ---------------------------------------
elif section == "5. Real-Time Sentiment Predictor":
    st.header("🔍 Real-Time Tweet Sentiment Classification")
    st.markdown("""
    Enter any tweet below to get an instant sentiment prediction (Positive or Negative).
    The model leverages both textual (TF-IDF) and engineered features to make its decision.
    """)

    user_input = st.text_area("Enter a tweet here:", height=120)

    if st.button("Classify Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please type a tweet to classify.")
        else:
            st.info(f"Processing your tweet: '{user_input[:100]}...'", icon="⚙️")

            # 1. Clean the tweet
            cleaned_tweet = clean_tweet(user_input)

            # 2. Calculate engineering features
            char_len = len(user_input)
            word_len = len(user_input.split())
            hashtag_count = len(extract_hashtags(user_input))
            mention_count = len(re.findall(r'@\w+', user_input))
            # Avoid division by zero if input is empty, though cleaned_tweet handles it too
            punct_density = sum(1 for c in user_input if c in '!?.') / max(len(user_input), 1)
            exclaim_count = user_input.count('!')
            has_url = int(bool(re.search(r'http\S+|www\S+', user_input)))

            eng_features = np.array([
                char_len, word_len, hashtag_count, mention_count,
                punct_density, exclaim_count, has_url
            ]).reshape(1, -1)

            # 3. Scale engineering features
            eng_features_scaled = scaler.transform(eng_features)

            # 4. Vectorize cleaned tweet and apply feature selection
            tfidf_features = vectorizer.transform([cleaned_tweet])
            tfidf_features_sel = selector.transform(tfidf_features)

            # 5. Combine all features
            final_features = sp.hstack([tfidf_features_sel, eng_features_scaled])

            # 6. Predict
            pred = model.predict(final_features)[0]
            pred_proba = model.predict_proba(final_features)[0]

            # Display
            st.subheader("Prediction Result:")
            if pred == 1:
                st.success(f"😊 **Positive Sentiment** (Confidence: {pred_proba[1]*100:.2f}%) detected!")
            else:
                st.error(f"😠 **Negative Sentiment** (Confidence: {pred_proba[0]*100:.2f}%) detected!")
            st.markdown(f"*Cleaned Text:* `{cleaned_tweet}`")

    st.markdown("---")
    st.markdown("Powered by Machine Learning models trained on the Sentiment140 dataset.")

# ---------------------------------------
# 7. Business Insights
# ---------------------------------------
elif section == "6. Business Insights":
    st.header("💡 Business Insights: Actionable Findings")
    st.markdown("""
    Leveraging the data mining and machine learning techniques applied in this project,
    here are some key actionable insights derived from the Sentiment140 dataset:
    """)

    st.subheader("Top 3 Actionable Findings:")

    st.markdown("---")
    st.markdown("#### Finding 1: Dominance of Contextual Language in Sentiment")
    st.markdown("""
    The SHAP analysis clearly showed that specific words like 'thanks', 'love', 'good' strongly push predictions towards positive sentiment,
    while words like 'sad', 'miss', 'hate', 'suck' are strong indicators of negative sentiment.
    **Actionable Insight**: For brands or social media managers, closely monitor the usage frequency and context of these highly influential words.
    Early detection of increasing negative terms related to a product/service can trigger proactive customer support or PR responses.
    Conversely, identifying frequently used positive terms can inform marketing campaigns and highlight successful aspects.
    """)

    st.markdown("---")
    st.markdown("#### Finding 2: Limited Influence of Structural Tweet Features on Sentiment")
    st.markdown("""
    The feature correlation heatmap indicated very weak correlations between engineered features like `char_len`, `word_len`,
    `hashtag_count`, `mention_count`, and `target` sentiment. This suggests that the length of a tweet or the presence of mentions/hashtags
    alone is generally not a strong predictor of its sentiment.
    **Actionable Insight**: While these features can be useful for other types of analysis (e.g., engagement),
    efforts to predict sentiment should primarily focus on the textual content itself rather than heavily relying on these structural features.
    This can streamline feature engineering for sentiment-specific models.
    """)

    st.markdown("---")
    st.markdown("#### Finding 3: Opportunity for Targeted User Engagement via Hashtag Analysis")
    st.markdown("""
    The hashtag sentiment profile revealed that certain hashtags are overwhelmingly positive (e.g., #followfriday, #musicmonday)
    or negative (e.g., #fail, #inaperfectworld).
    **Actionable Insight**: Social media strategists can leverage this.
    Engaging with users employing positive hashtags related to their brand can amplify positive sentiment.
    Conversely, monitoring negative hashtags can provide a pulse on trending dissatisfaction, allowing for
    direct intervention or competitive analysis. For example, if a competitor's product is trending with a high rate of negative hashtags,
    it presents an opportunity for a brand to highlight its own strengths.
    """)
