import numpy as np
import nltk
import string 
import random
f=open('./movie_lines.txt','r',errors='ignore')

raw_file=f.read()
#converting to lowercase
raw_file=raw_file.lower()

##using punkt tokenizer
nltk.download('punkt')

##downloading wordnet dict
nltk.download('wordnet')
nltk.download('omw-1.4')

sentence_tokens=nltk.sent_tokenize(raw_file)
word_tokens=nltk.word_tokenize(raw_file)

lemmer=nltk.stem.WordNetLemmatizer()
def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict=dict((ord(punct),None) for punct in string.punctuation)

def lem_Normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

##define greeting inputs
greet_inputs=('hello','hi','hey','htbot','how are you')
greet_responses=('Heyy','Hello','Hi There!','How can i help you')

def greet(sentence):
    for word in sentence.split():
        if word.lower in greet_inputs:
            return random.choice(greet_responses)
print(greet("hello in cs"))
        
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_msg):
    bot_response=''
    TfidVec=TfidfVectorizer(tokenizer= lem_Normalize,stop_words='english')
    tfidf=TfidVec.fit_transform(sentence_tokens)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf==0):
        bot_response=bot_response+'I am sorry , unable to understand you'
    else:
        bot_response=bot_response+sentence_tokens[idx]
    return bot_response

flag=True
print("Hello i am HT bolle to Himanshu Tushar ka baccha phele mujhe parnam karo fir agge ki baat karte h aur baat karne ka mann na ho to bye bol dena shamana mat")
while(flag):
    user_response=input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks'or user_response=='thank you'):
            flag=False
            print("HT ki taraf se bye m ek time pe ek hi wish poori karta hoon")
        else:
            if(user_response !=None):
                print(greet(user_response))
                print('HT :'+str(greet(user_response)))
            else:
                sentence_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                print("HT :",end='')
                print(response(user_response))
                sentence_tokens.remove(user_response)
    else:
        flag=False
        print("HT : Goodbye")