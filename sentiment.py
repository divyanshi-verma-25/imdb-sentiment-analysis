import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
csv_path = "/home/tiya/Downloads/archive/IMDB Dataset.csv"
df = pd.read_csv(csv_path)
print("🌸 Dataset loaded successfully!")

# 2. Text Preprocessing Function
def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Keep only letters
    text = text.lower()                     # Lowercase
    return text

print("🌸 Cleaning 50,000 reviews... (This might take around 10-15 seconds)")
df['cleaned_review'] = df['review'].apply(clean_text)

# 3. Split Data into Train & Test Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], 
    df['sentiment'], 
    test_size=0.2, 
    random_state=42
)
print(f"🌸 Data split: {len(X_train)} training items, {len(X_test)} testing items.")

# 4. Feature Extraction (TF-IDF)
print("🌸 Extracting features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train the Model
print("🌸 Training Logistic Regression classifier...")
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# 6. Evaluate Model
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("\n🌸 --- RESULTS --- 🌸")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Interactive Custom Testing
print("\n" + "="*40)
print("🌸 TEST THE MODEL WITH YOUR OWN REVIEWS! 🌸")
print("========================================")

while True:
    user_input = input("\nEnter a movie review (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("🌸 Goodbye!")
        break
        
    if not user_input.strip():
        continue
        
    # Clean and vectorize the input text
    cleaned_input = clean_text(user_input)
    input_tfidf = vectorizer.transform([cleaned_input])
    
    # Predict sentiment and probability
    prediction = model.predict(input_tfidf)[0]
    probabilities = model.predict_proba(input_tfidf)[0]
    
    # Get probability confidence based on class ordering
    pos_prob = probabilities[1] if model.classes_[1] == 'positive' else probabilities[0]
    neg_prob = 1 - pos_prob
    
    # Print the verdict
    emoji = "🍿" if prediction == "positive" else "👎"
    confidence = pos_prob if prediction == "positive" else neg_prob
    
    print(f"🌸 Verdict: {prediction.upper()} {emoji} (Confidence: {confidence * 100:.1f}%)")