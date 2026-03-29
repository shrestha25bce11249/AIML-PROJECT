# AIML-PROJECT

# 📧 Spam Email Detector Using AI

A machine learning project that automatically detects whether an email is **spam or normal (ham)** using Natural Language Processing (NLP) and the Naive Bayes algorithm — built completely in Python.

> Made by **Shrestha Gupta** | B.Tech CSE / AI-ML | 2025-26

---

## 🤔 What Does This Project Do?

You paste any email text into the terminal and the AI instantly tells you:

- Whether it is **SPAM** or **SAFE**
- The **confidence percentage** (e.g. 96.3% spam)

Example output:

```
  *** SPAM DETECTED ***
  email  : Congratulations you have won a free iPhone just cl...
  result : SPAM
  spam % : 96.3%
  safe % : 3.7%
```

```
  >>> looks safe to me
  email  : Hey can you send me the notes from todays class
  result : Normal / Ham
  spam % : 4.1%
  safe % : 95.9%
```

---

## 🧠 How Does the AI Work?

The project follows these steps:

```
Raw Email Text
      ↓
Clean the text (remove symbols, URLs, punctuation)
      ↓
TF-IDF Vectorization (convert words → numbers)
      ↓
Naive Bayes Model (trained on 30 labelled emails)
      ↓
Prediction: SPAM or HAM + confidence %
```

**Why Naive Bayes?**
It is the classic algorithm for spam detection — fast, lightweight, and works great even with small datasets. Gmail originally used something similar!

---

## 📁 Project Structure

```
spam-email-detector/
│
├── spam_detector_student.py   ← main Python file (run this)
└── README.md                  ← you are reading this
```

---

## ⚙️ Setup — How to Run It

### Step 1 — Make sure Python is installed

Open your terminal and type:

```bash
python3 --version
```

You should see something like `Python 3.10.x`. If not, download Python from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Install the required libraries

This project needs only 2 libraries. Run this in your terminal:

```bash
pip3 install scikit-learn numpy
```

> **Windows users** use `pip` instead of `pip3`
> **Mac users** use `pip3` and `python3`

If you get a "pip not found" error, try:

```bash
python3 -m pip install scikit-learn numpy
```

---

### Step 3 — Download the project file

Clone this repository:

```bash
git clone https://github.com/your-username/spam-email-detector.git
cd spam-email-detector
```

Or just download `spam_detector_student.py` directly and save it on your computer.

---

### Step 4 — Run the project

```bash
python3 spam_detector_student.py
```

> **Windows:** `python spam_detector_student.py`

---

## 🖥️ How to Use It

When you run the file, it will:

1. Load and clean the dataset automatically
2. Train the AI model (takes less than 1 second)
3. Test itself on 6 example emails and print results
4. Ask **you** to type any email to check

```
=== try it yourself! type any email below ===
(type 'quit' when you want to stop)

paste email here: _
```

Just paste any email text and press **Enter**. Type `quit` to exit.

---

## 🧪 Things to Try

Copy and paste these into the terminal to test it:

**Should be SPAM:**
```
Congratulations! You have won a free iPhone. Click here to claim your prize now!
```
```
Make $5000 weekly from home. No experience needed. Join free today!
```
```
Your account has been compromised. Verify your details immediately or be blocked.
```

**Should be SAFE:**
```
Hi, can you please send me the assignment before Friday evening?
```
```
Mom, I will be home for dinner by 7pm. Can you make rajma chawal today?
```
```
The meeting has been rescheduled to Monday at 11am. Please update your calendar.
```

---

## 📊 Model Performance

Trained and tested on a 30-email dataset (70% train, 30% test):

| Model | Accuracy |
|---|---|
| Naive Bayes ✅ | ~88% |
| Logistic Regression | ~85% |

Naive Bayes was chosen as the final model.

---

## 🛠️ Libraries Used

| Library | Purpose |
|---|---|
| `scikit-learn` | Machine learning models and TF-IDF |
| `re` | Text cleaning (built into Python, no install needed) |
| `numpy` | Numerical operations |

---

## ⚠️ Known Limitations

- Dataset is small (only 30 emails) — accuracy will improve with more data
- Model may struggle with very formal-sounding spam (e.g. phishing that mimics real bank emails)
- No GUI — runs only in the terminal for now

---

## 🔮 Future Plans

- [ ] Use the full [Kaggle SMS Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (5,572 emails)
- [ ] Add a simple web interface using Flask
- [ ] Try a BERT model for better accuracy
- [ ] Build a Chrome extension that checks emails inside Gmail

---

## 📚 What I Learned

- How TF-IDF converts words into numbers the AI can understand
- Why Naive Bayes is still one of the best algorithms for text classification
- The importance of train-test split (do NOT test on training data!)
- How to clean raw text before feeding it to a model

---

## 🙏 References

- [Scikit-learn Documentation](https://scikit-learn.org)
- [Kaggle SMS Spam Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Paul Graham — A Plan for Spam (2002)](http://paulgraham.com/spam.html)
- [GeeksforGeeks — TF-IDF Explained](https://www.geeksforgeeks.org/tf-idf-model-for-page-ranking/)
- Krish Naik NLP playlist on YouTube

---

## 📬 Contact

**Shrestha Gupta**
B.Tech CSE / AI-ML | 2025-26

Feel free to fork this repo and improve it!
