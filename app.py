from flask import Flask, request, jsonify
import numpy as np
import nltk
import string 
import random

app = Flask(__name__)

f = open('./data.txt','r',errors='ignore')
raw_file = f.read()
raw_file = raw_file.lower()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence_tokens = nltk.sent_tokenize(raw_file)
word_tokens = nltk.word_tokenize(raw_file)

lemmer = nltk.stem.WordNetLemmatizer()
def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_Normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))


def greet(sentence):
    greet_inputs = ('hello', 'hi', 'hey', 'htbot', 'how are you')
    greet_responses = ('Heyy', 'Hello', 'Hi There!', 'How can i help you')
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_msg):
    bot_response=''
    TfidVec=TfidfVectorizer(tokenizer=None, stop_words='english')
    tfidf=TfidVec.fit_transform(sentence_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if req_tfidf==0:
        bot_response=bot_response+'I am sorry , unable to understand you'
    else:
        bot_response=bot_response+sentence_tokens[idx]
    return bot_response

def bot(user_response):
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks'or user_response=='thank you'):
            return "Your's Welcome , bye"
        else:
            if greet(user_response)!=None:
                return greet(user_response)
            else:
                sentence_tokens.append(user_response)
                # word_tokens=word_tokens+nltk.word_tokenize(user_response)
                # final_words=list(set(word_tokens))
                res=response(user_response)
                sentence_tokens.remove(user_response)
                return res
    else:
        return "Goodbye"

@app.route('/chatbot', methods=['POST'])
def chatbot():
    flag=True
    if flag:
        user_message = request.json['user_message']
        bot_response = bot(user_message)
        if  bot_response == 'I am sorry , unable to understand you':
            bot_response = bot_response+"\n Can you suggest an answer for this? : "
            return jsonify({'bot_response': bot_response})
           
        return jsonify({'bot_response': bot_response})
    else:
        new_ans=request.json['usr_msg']
        with open('data.txt', 'a') as file:
            file.write(user_message + "\n")
            file.write( new_ans+ ".\n")
        bot_response='Thank you for feedback'
        flag=True
        return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=5050)
