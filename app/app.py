import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "sentimet_model.joblib"
TFIDF_PATH = BASE_DIR / "model" / "tfidf_vectorizer.joblib"
DATA_PATH = BASE_DIR / "data" / "IMDB Dataset.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .positive-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }

    .negative-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    return model, tfidf


model, tfidf = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATA_PATH)

    return df


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    An NLP-based Machine Learning application using
    TF-IDF and Logistic Regression.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎬 Sentiment Analyzer")

st.sidebar.info(
    """
    This application uses:

    • IMDb 50K Movie Reviews  
    • Text Cleaning  
    • TF-IDF Vectorization  
    • Logistic Regression  
    • Streamlit
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Sentiment Prediction",
        "📊 Model Performance",
        "📈 Model Comparison",
        "📚 About Project"
    ]
)


# ============================================================
# PAGE 1 — SENTIMENT PREDICTION
# ============================================================

if page == "🏠 Sentiment Prediction":

    st.subheader("📝 Enter Your Movie Review")

    review = st.text_area(
        "Movie Review",
        height=200,
        placeholder=(
            "Example: I really loved this movie. "
            "The story was amazing and the acting was fantastic."
        )
    )

    # Example reviews
    st.markdown("### 💡 Try an Example")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "😊 Positive Example",
            use_container_width=True
        ):

            review = (
                "This movie was absolutely fantastic. "
                "The acting was brilliant and the story was amazing."
            )

    with col2:

        if st.button(
            "😞 Negative Example",
            use_container_width=True
        ):

            review = (
                "This movie was terrible and boring. "
                "It was a complete waste of time."
            )


    st.markdown("---")


    # ========================================================
    # PREDICTION
    # ========================================================

    if st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    ):

        if review.strip() == "":

            st.warning(
                "⚠️ Please enter a movie review first."
            )

        else:

            # Clean review
            cleaned_review = clean_text(review)

            # Convert to TF-IDF
            review_tfidf = tfidf.transform(
                [cleaned_review]
            )

            # Prediction
            prediction = model.predict(
                review_tfidf
            )[0]

            # Probability
            probabilities = model.predict_proba(
                review_tfidf
            )[0]

            negative_probability = probabilities[0]
            positive_probability = probabilities[1]

            confidence = max(probabilities)


            st.markdown("---")

            st.subheader("📊 Prediction Result")


            # =================================================
            # SENTIMENT RESULT
            # =================================================

            if prediction == 1:

                st.success(
                    "😊 POSITIVE SENTIMENT"
                )

            else:

                st.error(
                    "😞 NEGATIVE SENTIMENT"
                )


            # =================================================
            # CONFIDENCE
            # =================================================

            st.write(
                f"### Confidence: {confidence * 100:.2f}%"
            )

            st.progress(
                float(confidence)
            )


            # =================================================
            # PROBABILITIES
            # =================================================

            st.subheader(
                "📈 Sentiment Probabilities"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "😞 Negative",
                    f"{negative_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "😊 Positive",
                    f"{positive_probability * 100:.2f}%"
                )


            # =================================================
            # PROCESSED TEXT
            # =================================================

            st.subheader(
                "🔧 Processed Text"
            )

            with st.expander(
                "View cleaned review"
            ):

                st.write(
                    cleaned_review
                )


# ============================================================
# PAGE 2 — MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Performance of the Logistic Regression model "
        "on the IMDb test dataset."
    )


    # Load dataset
    df = load_dataset()


    # Clean reviews
    with st.spinner(
        "Preparing test data..."
    ):

        df["clean_review"] = df["review"].apply(
            clean_text
        )


        # Convert labels
        df["sentiment"] = df["sentiment"].map(
            {
                "negative": 0,
                "positive": 1
            }
        )


        X = df["clean_review"]

        y = df["sentiment"]


        # Same split used during training
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )


        # TF-IDF transformation
        X_test_tfidf = tfidf.transform(
            X_test
        )


        # Prediction
        y_pred = model.predict(
            X_test_tfidf
        )


    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )


    # ========================================================
    # METRIC CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )


    st.markdown("---")


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    fig, ax = plt.subplots(
        figsize=(6, 5)
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Negative",
            "Positive"
        ]
    )


    disp.plot(
        ax=ax
    )


    ax.set_title(
        "Logistic Regression Confusion Matrix"
    )


    st.pyplot(
        fig
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader(
        "📖 Confusion Matrix Interpretation"
    )

    st.write(
        f"""
        **Correct Negative Predictions:** {cm[0][0]}

        **Incorrect Negative Predictions:** {cm[0][1]}

        **Incorrect Positive Predictions:** {cm[1][0]}

        **Correct Positive Predictions:** {cm[1][1]}
        """
    )


# ============================================================
# PAGE 3 — MODEL COMPARISON
# ============================================================

elif page == "📈 Model Comparison":

    st.title(
        "📈 Machine Learning Model Comparison"
    )

    st.write(
        "Comparison of the models trained during the project."
    )


    # Your measured results from the same test set
    comparison = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Naive Bayes",
                "Linear SVM"
            ],

            "Accuracy": [
                89.96,
                86.77,
                89.50
            ]
        }
    )


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "📊 Accuracy Comparison"
    )


    chart_data = comparison.set_index(
        "Model"
    )


    st.bar_chart(
        chart_data["Accuracy"]
    )


    st.info(
        """
        The accuracy values shown here are the results
        obtained during model training on the IMDb test set.

        Logistic Regression was selected for this application
        because it achieved the highest measured accuracy
        among these three models in this project.
        """
    )


# ============================================================
# PAGE 4 — ABOUT PROJECT
# ============================================================

elif page == "📚 About Project":

    st.title(
        "📚 About This Project"
    )


    st.write(
        """
        This is an end-to-end Machine Learning and NLP project
        for classifying movie reviews as positive or negative.
        """
    )


    st.subheader(
        "📌 Dataset"
    )

    st.write(
        """
        IMDb Dataset of 50K Movie Reviews.

        Total reviews: 50,000

        Positive reviews: 25,000

        Negative reviews: 25,000
        """
    )


    st.subheader(
        "🧠 Machine Learning Pipeline"
    )

    st.code(
        """
IMDb 50K Dataset
        ↓
Data Cleaning
        ↓
Train/Test Split
        ↓
TF-IDF Vectorization
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Logistic Regression
        ↓
Joblib Serialization
        ↓
Streamlit Application
        """,
        language="text"
    )


    st.subheader(
        "🛠️ Technologies Used"
    )

    st.write(
        """
        • Python

        • Pandas

        • NumPy

        • Scikit-learn

        • NLTK concepts / NLP preprocessing

        • TF-IDF

        • Logistic Regression

        • Naive Bayes

        • Linear SVM

        • Joblib

        • Streamlit

        • Matplotlib
        """
    )


    st.subheader(
        "🎯 Project Objective"
    )

    st.write(
        """
        The objective is to build a machine learning system
        that can understand the sentiment expressed in movie
        reviews and classify them as Positive or Negative.
        """
    )