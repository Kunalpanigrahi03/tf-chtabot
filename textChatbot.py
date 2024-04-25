from flask import Flask, render_template, request, jsonify
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import json
import random
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import re
from keras.models import load_model

app = Flask(__name__)

model = load_model('chatbot_model_v2.h5')
lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words_v2.pkl','rb'))
classes = pickle.load(open('classes_v2.pkl','rb'))  
listings = json.loads(open('dataset/listings.json').read())
intents = json.loads(open('dataset/dataset.json').read())

def get_bot_response(user_input):
    country_preference = None
    
    # Regular expression to extract country preference
    country_match = re.search(r'hotels? in ([\w\s]+)', user_input, re.IGNORECASE)
    if country_match:
        country_preference = country_match.group(1)
        print("Country preference extracted:", country_preference)
    
    filtered_listings = []
    if country_preference:
        filtered_listings = [listing for listing in listings if listing['country'].lower() == country_preference.lower()]
        print("Filtered listings:", filtered_listings)
    
    # Tokenize the user input
    sentence_words = nltk.word_tokenize(user_input)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

    # Predict intent using the model
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1

    p = np.array(bag)
    
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    
    if return_list:
        tag = return_list[0]['intent']
        for intent in intents['intents']:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])
                break
        else:
            result = "I'm not sure what you're asking for."
    else:
        result = "I'm not sure what you're asking for."
    
    return result, filtered_listings

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/tf-chatbot", methods=['POST'])  
def tf_chatbot():
    user_input = request.form['user_input']
    bot_response, country_listings = get_bot_response(user_input)
    return jsonify({'user_input': user_input, 'bot_response': bot_response, 'country_listings': country_listings})

if __name__ == "__main__":
    app.run()
