import pickle
import sys
import numpy as np

def predict_fake_news(news_text):
    # 1. Load the saved model (this is now the entire pipeline)
    try:
        with open('model.pkl', 'rb') as f:
            model_pipeline = pickle.load(f)
    except FileNotFoundError:
        print("Error: Saved model not found. Please run train_model.py first.")
        return None

    # 2. Get prediction and confidence score
    prediction = model_pipeline.predict([news_text])[0]
    probabilities = model_pipeline.predict_proba([news_text])[0]
    
    # Get the confidence for the specific prediction
    confidence = np.max(probabilities) * 100
    
    return prediction, confidence

if __name__ == "__main__":
    print("=== Fake News Detector ===")
    
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("\nEnter the news article text: ")
        
    if not text.strip():
        print("Empty text provided. Exiting.")
        sys.exit()

    result = predict_fake_news(text)
    
    if result:
        label, score = result
        print(f"\nPrediction: {label.upper()}")
        print(f"Confidence Level: {score:.2f}%")
        
        if score < 60:
            print("\nNOTE: The model is not very confident about this prediction.")
            print("Hint: Try using words that are already in the news.csv dataset.")
