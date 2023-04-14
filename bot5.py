import numpy as np
import nltk
import string 
import random
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

f=open('./data.txt','r',errors='ignore')

raw_file=f.read()
#converting to lowercase
raw_file=raw_file.lower()

##using punkt tokenizer
nltk.download('punkt')

##downloading wordnet dict
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

sentence_tokens=nltk.sent_tokenize(raw_file)
word_tokens=nltk.word_tokenize(raw_file)

lemmer=nltk.stem.WordNetLemmatizer()
def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict=dict((ord(punct),None) for punct in string.punctuation)

def lem_Normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

preprocessed_corpus = [lem_Normalize(sentence) for sentence in sentence_tokens]
model = Word2Vec(preprocessed_corpus, window=5, min_count=5, workers=4)

# Save the trained model to disk
model.save("word2vec.model")
##define greeting inputs
greet_inputs=('hello','hi','hey','htbot','how are you')
greet_responses=('Heyy','Hello','Hi There!','How can i help you')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)
    
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_input(input_text):
    # convert to lowercase
    input_text = input_text.lower()
    # remove punctuation
    input_text = input_text.translate(str.maketrans("", "", string.punctuation))
    # tokenize input text
    tokens = word_tokenize(input_text)
    # remove stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return tokens

# define function to generate response
def generate_response(input_text):
    # preprocess input text
    input_tokens = preprocess_input(input_text)
    # calculate WMD distance between input and each sentence in the corpus
    wmd_distances = []
    for sentence in sentence_tokens:
        s_t = preprocess_input(sentence)
        distance = model.wv.wmdistance(input_tokens, s_t)
        wmd_distances.append(distance)
    # find index of closest sentence
    closest_idx = np.argmin(wmd_distances)
    # return corresponding sentence
    return sentence_tokens[closest_idx]

talking=True
print("HT : Hello i am HT ")
while(talking):
    print("User :",end="")
    user_response=input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks'or user_response=='thank you'):
            talking=False
            print("Your's Welcome , bye")
        else:
            if greet(user_response)!=None:
                print('HT :'+(greet(user_response)))
            else:
                sentence_tokens.append(user_response)
                print("HT :",end='')
                bot_response=generate_response(user_response)
                if(bot_response == 'I am sorry , unable to understand you'):
                    print(bot_response)
                    new_ans = input(" Can you suggest an answer for this? : ")
                    with open('data.txt', 'a') as file:
                        file.write(user_response + "\n")
                        file.write(new_ans + ".\n")
                    print("HT : Thank you for your suggestion, I will remember that.")
                else:
                    print(bot_response)
                sentence_tokens.remove(user_response)
    else:
        talking=False
        print("HT : Goodbye")
