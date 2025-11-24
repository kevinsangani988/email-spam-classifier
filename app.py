import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = list(stopwords.words('english'))

tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def text_transform(text):
    text = text.lower()
    tokenised_word = word_tokenize(text)
    rest_words = ['subject', 'e', 'mail', 'n']
    tokenised_word = [word for word in tokenised_word if word not in rest_words]

    y=[]

    for i in tokenised_word:
        if i.isalnum():
            y.append(i)

    z = []

    for i in y:
        if i not in stop_words and i not in string.punctuation:
            z.append(i)
    Output = []
    for i in z:
        Output.append(ps.stem(i))
    return " ".join(Output)

st.title("Email Spam Classifier")

st.subheader("Enter the email content below:")
email_content = st.text_area("Email Content", height=300)

if st.button("Predict"):
    email_content = text_transform(email_content)
    email_content_tfidf = tfidf_vectorizer.transform([email_content])
    prediction = model.predict(email_content_tfidf)

    if prediction[0] == 1:
        st.success("This email is classified as SPAM.")
    else:
        st.success("This email is classified as NOT SPAM.")
