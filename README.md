# Fake News Detector

Hi! This is a project I built to learn how Machine Learning handles text data. It’s a simple "Fake News" detector that looks at a news headline and tries to guess if it's real or fake based on the words used.

I used **Python** and the **scikit-learn** library to make this happen.

## How I built it
I used a technique called **TF-IDF** to turn text into numbers that a computer can understand, and then trained a **Naive Bayes** model to find patterns. For example, the model learns that words like "investing" usually show up in real news, while words like "breaking" might show up more in fake stories.

### 📂 What's inside:
- `data/news.csv`: A small dataset I put together with 40 examples of real and fake news.
- `train_model.py`: This is the script I use to "teach" the model.
- `predict.py`: This is the script I use to test new headlines.
- `requirements.txt`: The libraries you need to install.

---

## 🚀 Getting it running

### 1. Install Python
First, you'll need to have Python installed on your computer. You can download it from [python.org](https://www.python.org/downloads/). During installation, make sure to check the box that says **"Add Python to PATH"**.

### 2. Install the libraries
Once you have Python, you'll need to install the specific tools I used. Open your terminal and run:
```bash
python -m pip install -r requirements.txt
```
this installs pandas and scikit-learn

### 2. Training the AI
Before it can predict anything, the AI needs to "study" my dataset. Run this:
```bash
python train_model.py
```
This will create a `model.pkl` file, which is basically the AI's "brain."

### 3. Testing a headline
Now you can try it out! You can either type a headline directly:
```bash
python predict.py "The government is investing in new public transportation."
```
Or just run `python predict.py` and it will ask you to type something.

---

## 🔍 What I learned (The Weaknesses)
This was a great learning project, but because I only used **40 examples** to train it, I noticed some funny mistakes the model makes. I call these its "weaknesses":

1.  **It loves specific keywords:** If I type `"breaking news the sun is hot"`, it says it's **FAKE**. This is because the word "breaking" was used in a lot of my fake news examples. It doesn't know what the "sun" is; it just sees the word "breaking" and gets suspicious!
2.  **It can be easily tricked:** If I type `"investing in rocks is the new deal"`, it says it's **REAL**. Why? Because the word "investing" is almost always in my real news examples.
3.  **Confidence scores:** I added a feature that shows how "confident" the AI is. If it hasn't seen the words before, you'll see a low confidence score (like 55%), which means it's basically just guessing.

### Next Steps
If I were to take this further, I'd want to use a massive dataset (like 10,000+ rows) and maybe a more advanced model like **BERT** so it can actually "understand" the sentences instead of just looking for keywords!
