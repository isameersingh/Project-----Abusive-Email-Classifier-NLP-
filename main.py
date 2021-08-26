import streamlit as st
from pickle import load
st.title("Welcome in Comment Classification")
txt=st.text_input("WRITE MESSAGE HERE")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
def text_transformation(txt):
    txt = txt.lower()
    txt = word_tokenize(txt)
    y = []
    for i in txt:
        if i.isalnum():
            y.append(i)
    txt = y[:]
    y.clear()
    for i in txt:
        if i not in stopwords.words('english'):
            y.append(i)
    txt = y[:]
    y.clear()
    for i in txt:
        y.append(ps.stem(i))
    txt = y[:]

    return " ".join(txt)
cln_txt=text_transformation(txt)
mnb = load(open('multinomial_model.pkl','rb'))
tfidf = load(open('tfidf.pkl','rb'))
vector_input=tfidf.transform([cln_txt])
result=mnb.predict(vector_input)
if(st.button("CLICK HERE")):
    if(result==0):
        st.title("Abusive")
    else:
        st.title("NonAbusive")