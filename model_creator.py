
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers.legacy import RMSprop


poetry_data_file = pd.read_csv('data/poetry_dataset.csv')  # loads the dataset

length_of_sequence = 40
step = 3

sentences = []
next_possible_characters = []

text = ''

# converts the dataset to a single string
for i in range(1, 500):
    text = text + poetry_data_file['Content'][i].strip().lower()

text = text.replace('\n', ' ')

# finds out all the unique characters in the text
possible_characters = sorted(set(text))

char_to_index = dict((char, index) for index, char in enumerate(possible_characters))
index_to_char = dict((index, char) for index, char in enumerate(possible_characters))


if __name__ == '__main__':

    # The code below generates the training data
    for i in range(0, len(text) - length_of_sequence, step):
        sentences.append(text[i: i + length_of_sequence])
        next_possible_characters.append(text[i + length_of_sequence])

    train_x = np.zeros((len(sentences), length_of_sequence, len(possible_characters)), dtype=np.bool_)
    train_y = np.zeros((len(sentences), len(possible_characters)), dtype=np.bool_)

    for i, sentence in enumerate(sentences):
        for j, character in enumerate(sentence):
            train_x[i, j, char_to_index[character]] = 1
        train_y[i, char_to_index[next_possible_characters[i]]] = 1

    # The model
    model = tf.keras.models.Sequential()

    model.add(LSTM(128, input_shape=(length_of_sequence, len(possible_characters))))
    model.add(Dense(len(possible_characters)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

    model.fit(train_x, train_y, batch_size=256, epochs=4)

    model.save('poetry_generator.model')
