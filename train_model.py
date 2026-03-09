import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_model():
    # 1. Load dataset
    print("Loading data...")
    df = pd.read_csv('data/news.csv')
    df.dropna(inplace=True)
    
    X = df['text']
    y = df['label']

    # 2. Create a Pipeline
    # This combines the "Translator" (TF-IDF) and "Brain" (Naive Bayes) into one object
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('nb', MultinomialNB())
    ])

    # 3. Train on the full dataset
    print("Training model...")
    model_pipeline.fit(X, y)

    # 4. Self-check accuracy
    y_pred = model_pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

    # 5. Save the entire pipeline
    print("\nSaving model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)
        
    print("Done! You can now run predict.py")

if __name__ == "__main__":
    train_model()
