import collections

import helper
import numpy as np
import project_tests as tests


from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

english_sentences = helper.load_data('small_vocab_en')
french_sentences = helper.load_data('small_vocab_fr')
print('Dataset Loaded')

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])
print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')



def tokenize(x):
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))


def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')
tests.test_pad(pad)
# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)
print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size) 

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')

embeddor=Embedding(199,100,input_length=15)
encoder=Bidirectional(GRU(100))
repeater=RepeatVector(21)
decoder=Bidirectional(GRU(100,return_sequences=True))
denser=TimeDistributed(Dense(344,activation='softmax'))

# def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
#     """
#     Build and train a bidirectional RNN model on x and y
#     :param input_shape: Tuple of input shape
#     :param output_sequence_length: Length of output sequence
#     :param english_vocab_size: Number of unique English words in the dataset
#     :param french_vocab_size: Number of unique French words in the dataset
#     :return: Keras model built, but not trained
#     """
#     # TODO: Implement
#     learning_rate = 1e-3
#     model = Sequential()
#     model.add(Bidirectional(GRU(128, return_sequences = True, dropout = 0.1), 
#                            input_shape = input_shape[1:]))
#     model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))
#     model.compile(loss = sparse_categorical_crossentropy, 
#                  optimizer = Adam(learning_rate), 
#                  metrics = ['accuracy'])
#     return model
# # tests.test_bd_model(bd_model)


# # TODO: Train and Print prediction(s)
# tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
# tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# bidi_model = bd_model(
#     tmp_x.shape,
#     preproc_french_sentences.shape[1],
#     len(english_tokenizer.word_index)+1,
#     len(french_tokenizer.word_index)+1)


# # preproc_french_sentences=preproc_french_sentences.reshape(68203, 21, 1)

# preproc_french_sentences=preproc_french_sentences[:68203]

# bidi_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)

# # Print prediction(s)
# print(logits_to_text(bidi_model.predict(tmp_x[:1])[0], french_tokenizer))

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    learning_rate = 0.005
    
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model
# tests.test_model_final(model_final)
print('Final Model Loaded')

def final_predictions(x, y, x_tk, y_tk):
    tmp_X = pad(preproc_english_sentences)
    model = model_final(tmp_X.shape,
                        preproc_french_sentences.shape[1],
                        len(english_tokenizer.word_index)+1,
                        len(french_tokenizer.word_index)+1)
    
    model.fit(tmp_X, preproc_french_sentences, batch_size = 1024, epochs = 5, validation_split = 0.2)
 
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'
    eng_sentence = ['new jersey is sometimes quiet during autumn','paris is usually wet during august',
    'he dislikes apples','india is never busy during autumn', 'she drives that little red truck ',
    'my least favorite fruit is the lime']
    for i in range(len(eng_sentence)):
    	temp_sentence = [x_tk.word_index[word] for word in eng_sentence[i].split()]
    	temp_sentence = pad_sequences([temp_sentence], maxlen=x.shape[-1], padding='post')
    	sentences = np.array([temp_sentence[0], x[0]])
    	predictions = model.predict(sentences, len(sentences))
    	print('src: '+eng_sentence[i])
    	print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    	print('\n')

preproc_french_sentences=preproc_french_sentences[:68203]
final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)