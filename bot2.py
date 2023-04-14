import numpy as np
import nltk
import string 
import random
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the data
with open('./data.txt', 'r', errors='ignore') as f:
    raw_data = f.read().lower()

# Tokenize the data
sentence_tokens = nltk.sent_tokenize(raw_data)
word_tokens = nltk.word_tokenize(raw_data)

# Remove punctuation and lemmatize the words
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmatizer = nltk.stem.WordNetLemmatizer()

def normalize(text):
    return [lemmatizer.lemmatize(token) for token in nltk.word_tokenize(text.lower().translate(remove_punc_dict))]

# Define greeting inputs and responses
greet_inputs = ('hello', 'hi', 'hey', 'htbot', 'how are you')
greet_responses = ('Heyy', 'Hello', 'Hi There!', 'How can i help you')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Prepare the input sequences
input_sequences = []
for sentence in sentence_tokens:
    seq = normalize(sentence)
    if seq:
        input_sequences.append(seq)

# Set the maximum number of words to consider
max_input_words = 5000

# Create the tokenizer
tokenizer = Tokenizer(num_words=max_input_words)
tokenizer.fit_on_texts(input_sequences)

# Get the input word index
target_word_to_index = tokenizer.word_index
target_index_to_word = tokenizer.index_word

# Convert input sequences to integer sequences
input_sequences = tokenizer.texts_to_sequences(input_sequences)

# Get the maximum input sequence length
max_input_seq_length = max(len(seq) for seq in input_sequences)

# Pad the input sequences
input_sequences = pad_sequences(input_sequences, maxlen=max_input_seq_length, padding='post')

# Define the encoder-decoder model

# Set the latent dimension for the encoder and decoder
latent_dim = 256

# Define the encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(input_dim=max_input_words, output_dim=latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Define the decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(input_dim=max_input_words, output_dim=latent_dim, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(max_input_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the full model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Define the encoder model (for inference)
encoder_model = Model(encoder_inputs, encoder_states)

# Define the decoder model (for inference)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(dec_emb_layer(decoder_inputs), initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Define function to preprocess input sequences
def preprocess_input_sequences(input_sequences):
    encoder_input_data = np.zeros((len(input_sequences), max_input_seq_length))
    for i, input_sequence in enumerate(input_sequences):
        encoder_input_data[i, :len(input_sequence)] = input_sequence
    return encoder_input_data

# Define function to generate response
def generate_response():
    # Take input from user
    print("you:",end='')
    input_sequence = input()

    # Encode the input sequence
    states_value = encoder_model.predict(preprocess_input_sequences([input_sequence]))

    # Initialize target sequence with start token
    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = target_word_to_index['<start>']

    # Generate response word by word
    stop_condition = False
    response = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_sequence] + states_value)

        # Get the next word with highest probability
        predicted_index = np.argmax(output_tokens[0, -1, :])
        predicted_word = target_index_to_word[predicted_index]

        # Append the predicted word to the response
        response += predicted_word + ' '

        # Exit condition: either hit max length or find stop token
        if (predicted_word == '<end>' or len(response) > max_input_seq_length):
            stop_condition = True

        # Update the target sequence
        target_sequence = np.zeros((1, 1))
        target_sequence[0, 0] = predicted_index

        # Update states
        states_value = [h, c]

    return response

generate_response()
