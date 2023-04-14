import openai
import nltk
import string 
import random

# Set up OpenAI API credentials
openai.api_key = "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"

# Load NLTK modules
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load movie lines corpus
with open('./movie_lines.txt', 'r', errors='ignore') as f:
    raw_file = f.read()

# Preprocess the corpus
raw_file = raw_file.lower()
sentence_tokens = nltk.sent_tokenize(raw_file)

# Define greeting inputs and responses
greet_inputs = ('hello', 'hi', 'hey', 'htbot', 'how are you')
greet_responses = ('Heyy', 'Hello', 'Hi There!', 'How can I help you')

# Define function to preprocess user input
lemmer = nltk.stem.WordNetLemmatizer()
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

# Define function to generate GPT-3 response
def generate_response(prompt):
    completions = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    return message.strip()

# Initialize chatbot
print("HT: Hello, I am HT. How can I assist you?")
while True:
    user_input = input("User: ").lower()
    if user_input == "bye":
        print("HT: Goodbye!")
        break
    elif user_input in greet_inputs:
        print("HT:", random.choice(greet_responses))
    else:
        # Preprocess user input
        user_input = lem_normalize(user_input)
        # Add user input to corpus
        sentence_tokens.append(" ".join(user_input))
        # Generate response using GPT-3
        prompt = " ".join(sentence_tokens[-3:])
        bot_response = generate_response(prompt)
        # Print response
        print("HT:", bot_response)
        # Remove user input from corpus
        sentence_tokens.remove(" ".join(user_input))
