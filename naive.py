import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Naive Bayes Spam Classifier", layout="centered")

st.title("📩 Naive Bayes Spam Classifier")
st.write("Enter a message to classify whether it is Spam or Not Spam.")
    
# Training Data
texts = [
    "Win money now",
    "Limited time offer",
    "Claim your prize",
    "Meeting at 10am",
    "Let's have lunch",
    "Project discussion tomorrow"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Spam, 0 = Not Spam

# Train Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# User Input
user_input = st.text_input("Enter your message")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        X_test = vectorizer.transform([user_input])
        prediction = model.predict(X_test)

        if prediction[0] == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT Spam.")