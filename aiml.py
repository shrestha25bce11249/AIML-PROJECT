
import re  # this is for cleaning text (removing symbols etc)
from sklearn.feature_extraction.text import TfidfVectorizer  # converts words to numbers
from sklearn.naive_bayes import MultinomialNB  # this is the main AI model i used
from sklearn.linear_model import LogisticRegression  # tried this too but NB was better
from sklearn.model_selection import train_test_split  # splits data into train and test
from sklearn.metrics import accuracy_score, classification_report  # to check how good my model is


# -------------------------------------------------------
# STEP 1 - my email dataset
# i wrote these myself based on real spam i have received
# 15 spam and 15 normal emails
# -------------------------------------------------------

emails = [

    # spam emails (the bad ones)
    ("Congratulations! You have won a $1,000,000 lottery prize. Click here to claim now!", "spam"),
    ("FREE Viagra! Buy now at 90% discount. Limited time offer. Click immediately!", "spam"),
    ("You are selected for a cash reward. Send your bank details to receive $5000 today.", "spam"),
    ("URGENT: Your account will be suspended. Verify your password now at this link.", "spam"),
    ("Make money fast! Work from home and earn $500 per day. No experience needed!", "spam"),
    ("Dear winner, you have been chosen. Claim your free iPhone 15 by clicking here.", "spam"),
    ("Get rich quick! Invest $100 and earn $10000 in 7 days. 100% guaranteed!", "spam"),
    ("Hot singles in your area are waiting for you. Click now for free access!", "spam"),
    ("You owe taxes. Pay immediately or face arrest. Call this number now.", "spam"),
    ("Buy cheap medication online. No prescription needed. Lowest prices guaranteed!", "spam"),
    ("WINNER ALERT: You have won our weekly draw. Reply with your name and address.", "spam"),
    ("Lose 30 pounds in 30 days with this one weird trick doctors hate!", "spam"),
    ("Your PayPal account is limited. Confirm your details or account will be closed.", "spam"),
    ("Free gift card worth $500 waiting for you. Complete a quick survey to claim.", "spam"),
    ("Earn extra income from home. Join thousands who already make $3000 weekly.", "spam"),

    # normal emails (ham - not spam)
    ("Hi, can we reschedule our meeting to Thursday at 3pm? Let me know if that works.", "ham"),
    ("Please find attached the project report for Q3. Let me know your feedback.", "ham"),
    ("Happy birthday! Hope you have a wonderful day with your family.", "ham"),
    ("The library books you borrowed are due next Monday. Please return them on time.", "ham"),
    ("Team lunch is confirmed for Friday at 1pm at the usual place. See you there!", "ham"),
    ("Your Amazon order has been shipped. Expected delivery is tomorrow by 6pm.", "ham"),
    ("Reminder: Doctor appointment tomorrow at 10:30am at City Hospital, Room 204.", "ham"),
    ("Thanks for your application. We will review it and get back to you within 5 days.", "ham"),
    ("The new software update is available. Please install it at your convenience.", "ham"),
    ("Can you send me the notes from yesterday's lecture? I missed the last part.", "ham"),
    ("Mom asked if you are coming home for the holidays. Let us know your plans.", "ham"),
    ("The electricity bill for this month is attached. Due date is 15th of this month.", "ham"),
    ("Great work on the presentation today! The client was really impressed.", "ham"),
    ("Your flight booking is confirmed. Check-in opens 24 hours before departure.", "ham"),
    ("I found a bug in the code we discussed. I have pushed a fix to the repository.", "ham"),

]

# separate the emails and labels into two lists
# i spent like 20 mins figuring out this list comprehension syntax lol
all_texts  = [item[0] for item in emails]
all_labels = [item[1] for item in emails]

print("=" * 55)
print("     MY SPAM EMAIL DETECTOR - AI PROJECT")
print("     by Shrestha Gupta")
print("=" * 55)

print("\ntotal emails i have:", len(emails))
print("spam count :", all_labels.count("spam"))
print("normal count:", all_labels.count("ham"))


# -------------------------------------------------------
# STEP 2 - cleaning the email text
# i learned that raw text has a lot of noise
# so we need to clean it before giving it to the model
# -------------------------------------------------------

def clean_email(text):
    # make everything lowercase first
    text = text.lower()

    # remove any website links
    text = re.sub(r"http\S+", "", text)

    # remove @ mentions
    text = re.sub(r"@\w+", "", text)

    # remove punctuation and numbers, keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# apply cleaning to all emails
cleaned_emails = [clean_email(t) for t in all_texts]

# just checking it works - i printed this during testing
# print(cleaned_emails[0])  # uncomment to see

print("\n[done] step 2 - cleaned all emails")


# -------------------------------------------------------
# STEP 3 - converting labels to numbers
# the model cant understand "spam" or "ham" as words
# so spam = 1 and ham = 0
# -------------------------------------------------------

# i used a simple if else for this, easy to understand
numeric_labels = []
for label in all_labels:
    if label == "spam":
        numeric_labels.append(1)
    else:
        numeric_labels.append(0)

print("[done] step 3 - labels converted (spam=1, ham=0)")


# -------------------------------------------------------
# STEP 4 - splitting data into training and testing
# 70% for training, 30% for testing
# random_state=42 just means the split is always the same
# i learned that 42 is kind of a meme in ML lol
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_emails,
    numeric_labels,
    test_size=0.3,       # 30% for testing
    random_state=42      # so results are same every time
)

print("[done] step 4 - data split")
print("       training emails:", len(X_train))
print("       testing emails :", len(X_test))


# -------------------------------------------------------
# STEP 5 - TF-IDF vectorization
# this converts words into numbers
# TF-IDF stands for Term Frequency Inverse Document Frequency
# basically it gives higher score to important/rare words
# words like "free" "win" "click" will score high in spam
# i watched 3 youtube videos to understand this properly
# -------------------------------------------------------

# max_features = only use top 300 most important words
# ngram_range = (1,2) means look at single words AND pairs of words
# eg: "click here" as a pair is more suspicious than just "click" alone
my_vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))

# fit on training data and transform it to numbers
X_train_numbers = my_vectorizer.fit_transform(X_train)

# only transform test data (not fit again - learned this the hard way)
X_test_numbers = my_vectorizer.transform(X_test)

print("[done] step 5 - words converted to numbers using TF-IDF")


# -------------------------------------------------------
# STEP 6 - training the AI models
# i tried two models to see which one is better
# Naive Bayes is the classic one for spam detection
# apparently gmail used something like this originally
# -------------------------------------------------------

# model 1 - naive bayes
# this works on probability - which words are more likely in spam vs normal
nb_model = MultinomialNB()
nb_model.fit(X_train_numbers, y_train)
nb_predictions = nb_model.predict(X_test_numbers)
nb_accuracy = accuracy_score(y_test, nb_predictions) * 100

# model 2 - logistic regression
# i tried this too but naive bayes did better on my data
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_numbers, y_train)
lr_predictions = lr_model.predict(X_test_numbers)
lr_accuracy = accuracy_score(y_test, lr_predictions) * 100

print("\n[done] step 6 - models trained!")
print("\n--- model comparison ---")
print(f"naive bayes accuracy     : {nb_accuracy:.1f}%")
print(f"logistic regression      : {lr_accuracy:.1f}%")
print("\ni am going with Naive Bayes since it did better :)")


# -------------------------------------------------------
# STEP 7 - checking how well the model actually works
# classification report shows precision recall and f1 score
# took me a while to understand what these mean
# precision = of all emails it called spam, how many were actually spam
# recall = of all actual spam emails, how many did it catch
# -------------------------------------------------------

print("\n--- full report for naive bayes ---")
print("-" * 45)
print(classification_report(
    y_test,
    nb_predictions,
    target_names=["Normal Email", "Spam Email"]
))
print("-" * 45)


# -------------------------------------------------------
# STEP 8 - the fun part! predicting new emails
# give it any email text and it tells you spam or not
# -------------------------------------------------------

def check_email(email_text):
    # clean the input first
    cleaned = clean_email(email_text)

    # convert to numbers using the same vectorizer
    as_numbers = my_vectorizer.transform([cleaned])

    # get prediction
    result = nb_model.predict(as_numbers)[0]

    # get confidence percentages
    confidence = nb_model.predict_proba(as_numbers)[0]
    spam_percent = confidence[1] * 100
    safe_percent  = confidence[0] * 100

    # print nicely
    print()
    if result == 1:
        print("  *** SPAM DETECTED ***")
    else:
        print("  >>> looks safe to me")

    # trim long emails for display
    display_text = email_text if len(email_text) <= 60 else email_text[:60] + "..."
    print(f"  email  : {display_text}")
    print(f"  result : {'SPAM' if result == 1 else 'Normal / Ham'}")
    print(f"  spam % : {spam_percent:.1f}%")
    print(f"  safe % : {safe_percent:.1f}%")
    print("  " + "-" * 48)


# testing with some examples i thought of
print("\n\n=== testing my model with example emails ===\n")

check_email("Congratulations you have won a free iPhone just click here to claim")
check_email("Hey can you send me the notes from todays class please")
check_email("FREE MONEY!! Make 500 dollars daily working from home guaranteed!")
check_email("Reminder that your assignment is due this Friday by 11:59pm")
check_email("Your bank account has been blocked verify your details immediately")
check_email("Are you free this weekend we are planning a movie night")


# -------------------------------------------------------
# STEP 9 - let the user type their own email to check
# this is the interactive part i am most proud of tbh
# -------------------------------------------------------

print("\n\n=== try it yourself! type any email below ===")
print("(type 'quit' when you want to stop)\n")

while True:
    user_email = input("paste email here: ").strip()

    # check if user wants to quit
    if user_email.lower() in ["quit", "exit", "q", "stop"]:
        print("\nokay bye! hope the project gets good marks lol")
        break

    # make sure they typed something
    if len(user_email) < 4:
        print("please type a longer email!")
        continue

    # run the prediction
    check_email(user_email)