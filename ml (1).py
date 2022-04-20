import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

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
  return allTokens

print(getTokens("www.google.com/ravi/kiran"))


df = pd.read_csv('malicious_phish.csv')
print(len(df))

dfx = df["url"]
data_x = np.array(dfx.iloc[:500000]) #500 is the best size so far

dfy = df["type"]
data_y = np.array(dfy.iloc[:500000])

for i in range(len(data_y)):
  if data_y[i] in ['defacement', 'malware', 'phishing']:
    data_y[i] = 'bad'
  else:
    data_y[i] = 'good'


# tf_vec = TfidfVectorizer(tokenizer=getTokens)
count_vec = CountVectorizer(tokenizer=getTokens)
# hash_vec = HashingVectorizer(norm = None, n_features = 3968)
# w2v_vec = word2vec.Word2Vec(sentences, workers = 1, size = 2, min_count = 1, window = 2, sg = 0)

#get a vector for each url but use our customized tokenizer
# def vectorize_vocabulary():
# X_tf = tf_vec.fit_transform(data_x)#(["www.google.com"])       #get the X vector from data_x
X_count = count_vec.fit_transform(data_x)#(["www.google.com"])
# X_hash = hash_vec.fit_transform(data_x)#(["www.google.com"])
# X_w2v = w2v_vec.fit_transform(data_x) # not used as it is commonly used to find similarity in words using cosine similarity and skip gram model

# vectorize_vocabulary()
Y = data_y

#reshaping all the vectors to match the label(Y)

#TFid
# X_tf.reshape(-1,1)
# # print(X_tf)
# print(X_tf.shape)

#count vec
X_count.reshape(-1,1)
# print(X_tf)
print(X_count.shape)

#hash vec
# X_hash.reshape(-1,1)
# # print(X_tf)
# print(X_hash.shape)
# # X.reshape(-1,1)

Y.reshape(-1,1)
print(Y.shape)
print("Vectorized words")

## *Logistic Regression*

### *Tfidf*

# X_train, X_test, Y_train, Y_test = train_test_split(X_tf, Y, test_size=0.3, random_state=42) #split into training and testing set
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# lgs = LogisticRegression() #using logistic regression
# lgs.fit(X_train, Y_train)
# print(lgs.score(X_test, Y_test)) #pring the score. It comes out to be 98%

### *Count Vectorizer*

X_train, X_test, Y_train, Y_test = train_test_split(X_count, Y, test_size=0.3, random_state=42) #split into training and testing set
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
lgs = LogisticRegression() #using logistic regression
lgs.fit(X_train, Y_train)
print(lgs.score(X_test, Y_test)) #pring the score. It comes out to be 98%

### *Hash Vectorizer*

# X_train, X_test, Y_train, Y_test = train_test_split(X_hash, Y, test_size=0.3, random_state=42) #split into training and testing set
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# lgs = LogisticRegression() #using logistic regression
# lgs.fit(X_train, Y_train)
# print(lgs.score(X_test, Y_test))

### *Model Predict*

def model_predict(input):
    X_predict = [input]
    X_predict = count_vec.transform(X_predict)
    X_predict.reshape(-1,1)
    # print(pd.DataFrame(X_predict.toarray()))
    y_Predict = lgs.predict(X_predict)
    # print(y_Predict) #printing predicted values
    return (y_Predict)
print(model_predict("www.google.com"))

pickle.dump(lgs, open('model_count.pkl', 'wb'))
pickle.dump(count_vec, open('vectorizer_count.pkl', 'wb'))