# IMDb Movie Reviews - Sentiment Analysis Pipeline

A lightweight Natural Language Processing (NLP) pipeline built from scratch to classify movie reviews as either **Positive** or **Negative** using the classical IMDb Dataset (50,000 reviews). 

The project leverages text preprocessing techniques, a **TF-IDF Vectorizer**, and a **Logistic Regression** classifier to achieve an optimal balance between fast training times and high classification accuracy.

---

## 🚀 Performance Metrics
* **Model Accuracy:** `88.95%`
* **Dataset Size:** 50,000 highly polar movie reviews (balanced)
* **Training Time:** ~11 seconds

---

## 🛠️ Features & Pipeline Architecture

1. **Text Preprocessing & Cleaning:**
   * Cleans raw, noisy text by stripping away HTML tags (like `<br />`).
   * Eliminates punctuation and digits, focusing purely on alphabetic text.
   * Standardizes text strings to lowercase.
2. **Feature Extraction:**
   * Converts textual data into numeric form using `TfidfVectorizer`.
   * Filters out standard English stop words.
   * Limits vocabulary size to the top 10,000 features to control dimensionality.
3. **Classification Model:**
   * Implements a binary `LogisticRegression` classifier suited for high-dimensional text matrices.
4. **Interactive CLI Inference:**
   * Includes a terminal loop allowing you to input live custom reviews to test model predictions and confidence scores in real-time.

---

## 📦 Requirements & Installation

### Setup Environment
Ensure your virtual environment is active, then install the necessary dependencies:
```bash
pip install pandas scikit-learn
