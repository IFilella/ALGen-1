import numpy as np
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup
import rdkit.Chem as Chem

global smiles_to_int

smiles_to_int = {"n": 0, "[": 1, "\\": 2, "E": 3, "H": 4, ")": 5, "B": 6, "9": 7, "2": 8, "]": 9, "7": 10, "!": 11,
 "t": 12, "s": 13, "o": 14, "c": 15, "K": 16, "-": 17, "/": 18, "l": 19, "A": 20, "r": 21, "@": 22, "C": 23, "=": 24,
 "6": 25, "N": 26, "L": 27, "a": 28, "5": 29, "S": 30, "T": 31, "#": 32, "+": 33, "P": 34, "i": 35, "(": 36, "8": 37,
 "1": 38, "I": 39, "e": 40, "O": 41, "3": 42, "F": 43, "4": 44, ".": 45, 'Z':46, 'M':47, 'g':48, 'R':49, 'p':50}

global int_to_smiles

int_to_smiles = {"0": "n", "1": "[", "2": "\\", "3": "E", "4": "H", "5": ")", "6": "B", "7": "9", "8": "2", "9": "]",
 "10": "7", "11": "!", "12": "t", "13": "s", "14": "o", "15": "c", "16": "K", "17": "-", "18": "/", "19": "l", "20": "A",
 "21": "r", "22": "@", "23": "C", "24": "=", "25": "6", "26": "N", "27": "L", "28": "a", "29": "5", "30": "S", "31": "T",
 "32": "#", "33": "+", "34": "P", "35": "i", "36": "(", "37": "8", "38": "1", "39": "I", "40": "e", "41": "O", "42": "3",
 "43": "F", "44": "4", "45":".", "46": "Z", '47':'M', '48':'g', '49':"R", '50':'p'}

global amino_to_int

amino_to_int = {'C': 0,'D': 1, 'S': 2, 'Q': 3, 'K': 4, 'I': 5, 'P': 6, 'T': 7, 'F': 8, 'N': 9, 'G': 10, 'H': 11, 'L': 12, 
 'R': 13, 'W': 14, 'A': 15, 'V': 16, 'E': 17, 'Y': 18, 'M': 19, 'U': 20, 'X': 21, '!': 22}

global int_to_amino

int_to_amino = {'0': 'C', '1': 'D', '2': 'S', '3': 'Q', '4': 'K', '5': 'I', '6': 'P', '7': 'T', '8': 'F', '9': 'N', '10': 'G',
 '11': 'H', '12': 'L', '13': 'R', '14': 'W', '15': 'A', '16': 'V', '17': 'E', '18': 'Y', '19': 'M', '20': 'U', '21': 'X', '22': '!'}


def length_scanner(file):
    '''
    Scan smiles file and compute mean and median compund length.
    '''
    lengths = []
    with open(file) as f:
        for line in f:
            lengths.append(len(line))

    num_char = 2 * np.median(lengths)

    print("Mean length: ", np.mean(lengths))
    print("Most common length: ", np.median(lengths))

    return num_char


def load_data(data, max_len):
    '''
    Load smiles from file into numpy array filtering for max length.
    '''
    with open(data, 'r') as f:
        smiles = [ r.rstrip() for r in f if len(r) <= max_len]

    return np.array(smiles)

def smiles_cleaning(data):
    '''
    Clean SMILES input dataset from salts and charges.
    '''

    clean_smiles = []

    remover = SaltRemover()

    for smiles in data:
        try:
            m = Chem.MolFromSmiles(smiles)
            a = remover.StripMol(m, dontRemoveEverything=True)
            e = Cleanup(a)
            clean_smiles.append(Chem.MolToSmiles(e))
        except:
            pass

    return np.array(clean_smiles)

def seq_cleaning(data):
    '''
    Prepare sequences. 
    '''

    seq_pack = []

    with open(data, 'r') as f:
        sequences = [r.rstrip() for r in f]

    for sequence in sequences:
        seq_pack.append(sequence)


    return np.array(seq_pack), max([len(seq) for seq in seq_pack])

def one_hot_encoder(smiles, max_len, num_char):
    '''
    Encode smiles characters to 1|0 arrays of an array based on smiles to int
    diccionary.
    '''
    one_hot = np.zeros((smiles.shape[0], int(max_len), num_char), dtype=np.int8)

    for i, smile in enumerate(smiles):
        try:
            one_hot[i, 0, smiles_to_int["!"]] = 1
            for j, c in enumerate(smile):
                one_hot[i, j+1, smiles_to_int[c]] = 1
            one_hot[i, len(smile)+1:, smiles_to_int["E"]] = 1
        except:
            pass

    return one_hot[:,0:-1,:], one_hot[:,1:,:]

def one_hot_decoder(smile):
    '''
    Decode function based on the same rules as the encoder. For a single smile.
    '''
    return  "".join([int_to_smiles[str(np.argmax(char))] for char in smile])

def protein_one_hot_encoder(sequences, max_len, num_char):
    '''
    Encode protein residues to 1|0 arrays of an array based on protein to int
    diccionary.
    '''
    one_hot = np.zeros((sequences.shape[0], int(max_len), num_char), dtype=np.int8)

    for i, sequence in enumerate(sequences):
        try:
            one_hot[i, 0, amino_to_int["!"]] = 1
            for j, c in enumerate(sequence):
                one_hot[i, j+1, amino_to_int[c]] = 1
            one_hot[i, len(sequences)+1:, amino_to_int["X"]] = 1
        except:
            pass

    return one_hot[:,0:-1,:], one_hot[:,1:,:]

def protein_one_hot_decoder(sequence):
    '''
    Decode function based on the same rules as the encoder. For a single sequence.
    '''
    return  "".join([int_to_amino[str(np.argmax(amino))] for amino in sequence])


class Data_Generator(Sequence):
    '''
    Class with instances to generate input data for the VAE-LSTM.
    '''
    def __init__(self, input_data, labels, batch_size):
        self.input_data, self.labels = input_data, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.input_data) / float(self.batch_size)))

    def __getitem__(self, idx):
        x = self.input_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x, batch_y = np.array(x), np.array(y)

        return [batch_x, batch_x], batch_y
