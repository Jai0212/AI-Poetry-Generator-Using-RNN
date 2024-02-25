
from random import randint
import numpy as np
from tensorflow.keras.models import load_model
from model_creator import text, length_of_sequence, possible_characters, char_to_index, index_to_char


model = load_model('poetry_generator.model')  # loads the model


def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


# generates the poetry
def generate_poetry(length, temperature):

    start = randint(0, len(text) - length_of_sequence - 1)
    generated_poetry = ''

    sentence_local = text[start: start + length_of_sequence]
    generated_poetry += sentence_local

    for _ in range(length):
        train_x_local = np.zeros((1, length_of_sequence, len(possible_characters)))

        for j_local, character_local in enumerate(sentence_local):
            train_x_local[0, j_local, char_to_index[character_local]] = 1

        prediction = model.predict(train_x_local, verbose=0)[0]
        next_index = sample(prediction, temperature)
        next_char = index_to_char[next_index]

        generated_poetry += next_char
        sentence_local = sentence_local[1:] + next_char

    return generated_poetry


print(generate_poetry(300, 0.8))
# first parameter: length of poetry
# second parameter: creativity index (greater than 0, less than 1)
# works best with a creativity index of 0.8
