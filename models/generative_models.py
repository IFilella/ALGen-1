import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.backend import clear_session
import gc

from rdkit import Chem

from data.data_processing import int_to_smiles, smiles_to_int, amino_to_int, int_to_amino
from utils.utils import sample_with_temp

import numpy as np
import pdb

def lstm_model(X, y):
    '''
    Encoder/decoder architecture using LSTMs.
    Encoder: Input -> LSTM (hidden, current states) -> Fully connected layer (bottle neck)
    Decoder: (hidden, current states) -> Fully connected layers -> LSTM -> Fully connected layer (output)
    '''
    print(X.shape)
    print(y.shape)
    # Encoder
    enc_input = Input(shape=(X.shape[1:]))
    _, state_h, state_c = LSTM(256, return_state=True)(enc_input)
    states = Concatenate(axis=-1)([state_h, state_c])
    bottle_neck = Dense(128, activation='relu')(states)

    # Decoder
    state_h_decoded = Dense(256, activation='relu')(bottle_neck)
    state_c_decoded = Dense(256, activation='relu')(bottle_neck)
    encoder_states = [state_h_decoded, state_c_decoded]
    dec_input = Input(shape=(X.shape[1:]))
    dec1 = LSTM(256, return_sequences=True)(dec_input, initial_state=encoder_states)
    try:
      output = Dense(y.shape[2], activation='softmax')(dec1)
    except:
      output = Dense(y.shape[1], activation='softmax')(dec1)

    model = Model(inputs=[enc_input, dec_input], outputs=output)

    return model


def sample_smiles(gen_model, latent_to_states_model, latent, n_vocab, sampling_temp):

    states = latent_to_states_model.predict(latent.reshape(1, -1))
    gen_model.layers[1].reset_states(states=[states[0], states[1]])

    startidx = smiles_to_int["!"]
    samplevec = np.zeros((1,1,n_vocab))
    samplevec[0,0,startidx] = 1
    sequence = ""

    for i in range(n_vocab):
        preds = gen_model.predict_on_batch(samplevec)[0][-1]
        if sampling_temp == 1.0:
          sampleidx = np.argmax(preds)
        else:
          sampleidx = sample_with_temp(preds, sampling_temp)
        samplechar = int_to_smiles[str(sampleidx)]
        if samplechar != "E":
            sequence += samplechar
            samplevec = np.zeros((1,1,n_vocab))
            samplevec[0,0,sampleidx] = 1
        else:
            break
    return sequence

def sample_sequences(gen_model, latent_to_states_model, latent, n_vocab, sampling_temp):

    states = latent_to_states_model.predict(latent)
    gen_model.layers[1].reset_states(states=[states[0], states[1]])

    startidx = amino_to_int["!"]
    samplevec = np.zeros((1,1,n_vocab))
    samplevec[0,0,startidx] = 1
    sequence = ""

    for i in range(n_vocab):
        preds = gen_model.predict(samplevec)[0][-1]
        if sampling_temp == 1.0:
          sampleidx = np.argmax(preds)
        else:
          sampleidx = sample_with_temp(preds, sampling_temp)
        samplechar = int_to_amino[str(sampleidx)]
        if samplechar != "E":
            sequence += samplechar
            samplevec = np.zeros((1,1,n_vocab))
            samplevec[0,0,sampleidx] = 1
        else:
            break
    return sequence

def generate(gen_model, latent_to_states_model, latent_seed, sampling_temp, n_vocab, scale, quant, protein=None):

    samples, mols = [], []
    not_valid = 0

    for i in range(quant):
        latent_vec = latent_seed + scale*(np.random.randn(latent_seed.shape[1]))
        out = sample_smiles(gen_model, latent_to_states_model, latent_vec, n_vocab, sampling_temp)

        if not protein:
          mol = Chem.MolFromSmiles(out)

          if mol:
            mols.append(mol)
            samples.append(out)
          else:
            not_valid += 1

        else:
          out = sample_sequences(gen_model, latent_to_states_model, latent_vec, n_vocab, sampling_temp)
          samples.append(out)

        # clear the memory every 100 iterations
        if i % 100 == 0:
          gc.collect()

    print("\nTried to generate {} compounds and {} were not valid.".format(quant, not_valid))

    return mols, samples, not_valid

