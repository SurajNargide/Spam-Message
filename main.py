import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv(r'static\SMSSpamCollection.csv', encoding='unicode_escape')
data = data.dropna()
data['text'] = data['text'].astype(str)
data['text'] = [x.lower() for x in data['text']]
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return nopunc


data['text'] = data['text'].apply(text_process)
X = data['text']
y = data['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model_1 = RandomForestClassifier()
text_clv = Pipeline([('Tfidf',TfidfVectorizer()),('model_1',RandomForestClassifier(n_estimators=100,max_features=10))])
text_clv.fit(X_train,y_train)

predict = text_clv.predict(["['doctor', 'mbbs', 'student', 'found', 'shot', 'dead', 'hostel']"])
print(predict)

pickle.dump(text_clv, open("model.pkl",'wb'))