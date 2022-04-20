import streamlit as st
import time as time
import base64
import pickle
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
import webbrowser
import pandas as pd


def getTokens(link):
  link = link.replace("www.", "")
  link = link.replace(".com", "")
  print(link)
  tokensBySlash = str(link).split('/') #get tokens after splitting by slash
  allTokens = []
  for i in tokensBySlash:
    tokens = str(i).split("-") #get tokens after splitting by dash
    tokensByDot = []
    for j in range(0,len(tokens)):
      tempTokens = str(tokens[j]).split('.') #get tokens after splitting by dot
      tokensByDot = tokensByDot + tempTokens
    allTokens = allTokens + tokens + tokensByDot
  allTokens = list(set(allTokens)) #remove redundant tokens
  # if "com" or "www" in allTokens:
  #   allTokens.remove('com') #removing .com since it occurs a lot of times and it should not be included in our features
  #   allTokens.remove('www')
  return allTokens

print(getTokens("www.google.com/ravi/kiran"))




model = pickle.load(open('model_count.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_count.pkl', 'rb'))
st.cache(allow_output_mutation=True)
main_bg ="web design.jpg"
main_bg_ext = "jpg"

side_bg = "web design.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Malicious website detector")
#st.caption("this website helps us detecting whether website has malcious content or not")
st.subheader("Enter the url")

web_box=st.text_input("")
# st.write(web_box)
if (web_box != "" or st.button("CHECK")):
    checking = st.text("checking...")
    progress=st.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress.progress(i+1)
    
    checking.empty()
    progress.empty()

    # hash_vec = HashingVectorizer(norm = None, n_features = 3968)
    # tf_vec = TfidfVectorizer(tokenizer=getTokens)

    X_predict = [web_box]


    print("before:",X_predict)
    X_predict = vectorizer.transform(X_predict)
    print("after:",X_predict)
    X_predict.reshape(-1,1)

    reply = model.predict(X_predict)
    
    st.write("checking completed")
    print("reply:",reply)   
    if(reply == ["good"]):
        st.success("Website is safe")
    else:
        st.error("Malicious content detected!!!")
    
    # lr_button = st.button("Learn more")
    st.write(f'''
        <a target="_blank" href="https://www.ecsu.edu/administration/information-technology/resources/infosec/cyber-security-awareness-for-students.html">
            <button style = "background-color:#0068c9; border-radius:5px;">
                Learn more about web security
            </button>
        </a>
        ''',
        unsafe_allow_html=True
    )
