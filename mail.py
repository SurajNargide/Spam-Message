from flask import Flask, render_template, request, url_for
import pandas as pd
import pickle
import string
import nltk.sentiment
nltk.download('vader_lexicon')


mail = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@mail.route("/", methods = ["POST", "GET"])
def home():
    return render_template('home.html')

@mail.route("/index", methods = ["POST", "GET"])
def index():
    text = request.form['text']
    lowered_text = [x.lower() for x in text]

    def text_process(mess):
        nopunc = [char for char in mess if char not in string.punctuation]

        nopunc = ''.join(nopunc)

        return nopunc

    processed_data = text_process(lowered_text)
    prediction = model.predict([processed_data])
    print(prediction)
    if prediction == 'ham':
        y = 'Okay'
    else:
        y = 'Spam'

    def get_sentiment(text):
        from nltk.sentiment import SentimentIntensityAnalyzer
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        analysis = sia.polarity_scores(text)

        if analysis['compound'] > 0:
            return 'Positive'
        elif analysis['compound'] < 0:
            return 'Negative'
        else:
            return 'Neutral'

    sentiment = str(get_sentiment(text))
    return render_template('index.html', y = y, sentiment = sentiment)


if __name__ == "__main__":
    mail.run(debug= True)