
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from random import choices

def read_and_process_text():
    filename = "alice_in_wonderland.txt"
    raw_text = open(filename).read().lower()
    translation_table = dict.fromkeys(map(ord, '\n\r'), ' ')
    raw_text = raw_text.translate(translation_table)
    translation_table = dict.fromkeys(map(ord, '\"\'()*-03:;[]_`'), None)
    raw_text = raw_text.translate(translation_table)
    raw_text = " ".join(raw_text.split())
    letters = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(letters))
    one_hot = np_utils.to_categorical([char_to_int[i] for i in letters]).tolist()
    return (one_hot, char_to_int, letters, raw_text)

def create_dataset(one_hot, char_to_int, letters, raw_text, seq_length):
    dataX = list()
    dataY = list()
    for i in range(0, len(raw_text) - seq_length):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([one_hot[char_to_int[char]] for char in seq_in])
        dataY.append(one_hot[char_to_int[seq_out]])
    print("Total Characters: ", len(raw_text))
    print("Total Vocab: ", len(letters))
    print("Total Sequents: ", len(dataX))
    return (np.array(dataX), np.array(dataY))

def get_model(X, Y):
  model = Sequential()
  model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
  model.add(Dropout(0.2))
  model.add(Dense(Y.shape[1], activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  model.summary()
  return model

def run_model(model, X, Y):
    filepath = "run/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, Y, epochs=50, batch_size=64, callbacks=callbacks_list)

def load_model(filename, model):
  model.load_weights(filename)
  model.compile(loss='categorical_crossentropy', optimizer='adam')

def LSTM_gen(string, num_chars_to_predict, seq_length, letters,
                    one_hot, char_to_int, model):
  if (seq_length > len(string)):
      print("Cannot predict(too short init string)")
      return string
  last_string = string[-seq_length:]
  current_window = [one_hot[char_to_int[ch]] for ch in last_string]
  for i in range(num_chars_to_predict):
    x = np.array([current_window])
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = letters[index]
    string += result
    print(result, end='')
    current_window.append(one_hot[index])
    current_window = current_window[1:len(current_window)]
    #next_str = [letters[np.argmax(i)] for i in current_window]
    #print(''.join(next_str))
  return string

(one_hot, char_to_int, letters, text) = read_and_process_text()

seq_length = 100
[X,Y] =  create_dataset(one_hot, char_to_int, letters, text, seq_length)

model = get_model(X,Y)
run_model(model, X, Y)

string = "alica was the main character and never give up because of difficulties on her way and others really admired her because of her charecter. "
load_model("weights-17.hdf5", model)
num_chars_to_predict = 1000
gen_string = LSTM_gen(string, num_chars_to_predict, seq_length, letters, one_hot, char_to_int, model)
print(gen_string)

def markov_chain(seq_length, text, letters):
    table = {}
    for i in range(0, len(text) - seq_length):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        if (not (seq_in in table)):
            cur_letters = {}
            for ch in letters:
                cur_letters[ch] = 0
            table[seq_in] = cur_letters
        table[seq_in][seq_out] += 1
    for seq in table:
        tt = sum(table[seq].values())
        for ch in table[seq]:
            table[seq][ch] /= tt
    return table

def markov_gen(string, table, num_chars_to_predict, seq_length, letters):
    if (seq_length > len(string)):
        print("Cannot predict(too short init string")
        return string
    last_string = string[-seq_length:]
    for i in range(num_chars_to_predict):
        if not(last_string in table):
            result = choices( population=letters, weights=[1 / len(letters) for i in range(len(letters))], k=1)
        else:
            result = choices(population= letters, weights=[table[last_string][ch] for ch in letters], k = 1)
        string += result[0]
        last_string = string[-seq_length:]
    return string

markov_seq = 20
table = markov_chain(markov_seq, text, letters)
markov_gen_string =  markov_gen(string, table, num_chars_to_predict, seq_length, letters)
print(markov_gen_string)