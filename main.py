import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
import webbrowser

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


df = pd.read_csv('malicious_phish.csv')
print(len(df))

dfx = df["url"]
data_x = np.array(dfx.iloc[:100000]) #500 is the best size so far


dfy = df["type"]
data_y = np.array(dfy.iloc[:100000])


for i in range(len(data_y)):
  if data_y[i] in ['defacement', 'malware', 'phishing']:
    data_y[i] = 'bad'
  else:
    data_y[i] = 'good'


tf_vec = TfidfVectorizer(tokenizer=getTokens)

X_tf = tf_vec.fit_transform(data_x)

Y = data_y

#TFid
X_tf.reshape(-1,1)
# print(X_tf)
print(X_tf.shape)

Y.reshape(-1,1)
print(Y.shape)
print("Vectorized words")

#train model
X_train, X_test, Y_train, Y_test = train_test_split(X_tf, Y, test_size=0.3, random_state=42) #split into training and testing set
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
lgs = LogisticRegression() #using logistic regression
lgs.fit(X_train, Y_train)
print(lgs.score(X_test, Y_test)) #pring the score. It comes out to be 98%


def model_predict(input):
    X_predict = [input]
    X_predict = tf_vec.transform(X_predict)
    X_predict.reshape(-1,1)
    # print(pd.DataFrame(X_predict.toarray()))
    y_Predict = lgs.predict(X_predict)
    # print(y_Predict) #printing predicted values
    return (y_Predict)
print(model_predict("www.google.com"))



##########################################  UI    ####################################################################################


import streamlit as st
import time as time
import base64
import pickle
import numpy as np

#model = pickle.load(open('model.pkl', 'rb'))
#vectorizer = pickle.load(open('vectorize.pkl', 'rb'))
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
st.title("Fake website detector")
#st.caption("this website helps us detecting whether website has malcious content or not")
st.subheader("Enter the url")

web_box=st.text_input("")
#st.write(web_box)
if (web_box != "" or st.button("CHECK")):
    st.write("checking.....")
    progress=st.progress(0)
    for i in range(100):
        time.sleep(0.005)
        progress.progress(i+1)
    progress.empty()
    X_predict = web_box
    print("X_predict", X_predict)
    reply = model_predict(X_predict)
    print("reply:",reply)
    st.write("checking completed") 
    # st.write(reply)   
    if(reply == ['good']):
        st.success("Website is safe")
    else:
        st.error("Malicious content detected")
        lr_button=st.button("Learn more")

    st.write(f'''
        <a target="_blank" href="https://www.ecsu.edu/administration/information-technology/resources/infosec/cyber-security-awareness-for-students.html">
            <button>
                Learn More
            </button>
        </a>
        ''',
        unsafe_allow_html=True
    )  

