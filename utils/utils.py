import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem import AllChem, DataStructs
from . import sascorer

import matplotlib.pyplot as plt
import seaborn as sns

import selfies as sf
import deepsmiles as ds

from collections import Counter


def write_report(train_size, val_size, test_size, test_loss, test_acc, epochs, batch_size, sampling_temp, quant, not_valid, name, outdir):
    with open(outdir + "/report_" + name + '.out', 'w+') as r:
        r.write('#######################################\n')
        r.write('Report for generative run ' + name + '\n')
        r.write('#######################################\n')
        r.write('Model trained for: ' + str(epochs)+ ' epoch' + '\n')
        r.write('Chosen batch size: ' + str(batch_size)+ '\n')
        r.write('Sampling temperature: ' + str(sampling_temp)+ '\n')
        r.write('#######################################'+ '\n')
        r.write('Train size: ' + str(train_size)+ '\n')
        r.write('Validation size: ' + str(val_size)+ '\n')
        r.write('Test size: ' + str(test_size)+ '\n')
        r.write('#######################################'+ '\n')
        r.write('Test loss: ' + str(test_loss)+ '\n')
        r.write('Test accuracy: ' + str(test_acc)+ '\n')
        r.write('#######################################'+ '\n')
        r.write('Tried to generate ' + str(quant) + ' compounds and ' + str(not_valid) + ' were not valid'+ '\n')
        if not_valid > 0:
            success_rate = (1 - float(not_valid)/(float(quant))) * 100
        else:
            success_rate = 0.0
        r.write('Success rate: ' + str(success_rate) + '%')

def to_gpu():
    try:
        print('Num GPUs available {}'.format(tf.config.experimental.list_physical_devices('GPU')))

    except:
        print('Tensorflow did not find any NVIDIA GPU or Tensorflow-GPU is not installed.')

def plot_acc_loss(history, name, output_dir):
    fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

    axis1.plot(history.history["acc"], label='Train', linewidth=3)
    axis1.plot(history.history["val_acc"], label='Validation', linewidth=3)
    axis1.set_title('Model accuracy', fontsize=16, color="white")
    axis1.set_ylabel('accuracy')
    axis1.set_xlabel('epoch')
    axis1.legend(loc='lower right')

    axis2.plot(history.history["loss"], label='Train', linewidth=3)
    axis2.plot(history.
history["val_loss"], label='Validation', linewidth=3)
    axis2.set_title('Model loss', fontsize=16, color="white")
    axis2.set_ylabel('loss')
    axis2.set_xlabel('epoch')
    axis2.legend(loc='upper right')

    plt.savefig(output_dir + '/' + name + "_accuracy_loss.png", format='png')

    return 0


def sample_with_temp(preds, sampling_temp):

    streched = np.log(preds) / sampling_temp
    streched_probs = np.exp(streched) / np.sum(np.exp(streched))

    return np.random.choice(range(len(streched)), p=streched_probs)


def compute_tanimoto_init(specific_set):
    '''
    '''

    bulk = []
    fps = []

    with open(specific_set, 'r') as f:
        for i in f:
            bulk.append(Chem.MolFromSmiles(i))

    for x in bulk:
        try:
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4)
            fps.append(fpgen.GetSparseCountFingerprint(x))
        except:
            pass

    return fps


def tanimoto2(compound, specific_set_fps):
    '''
    '''
    comp_mol = Chem.MolFromSmiles(compound)

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=4)
    comp_fps = fpgen.GetSparseCountFingerprint(comp_mol)

    sims = DataStructs.BulkTanimotoSimilarity(comp_fps, specific_set_fps)

    return np.mean(sims), np.max(sims)

def compute_properties(smi):
    '''
    '''
    print('smiles: ', smi)
    mol = Chem.MolFromSmiles(smi)

    sa = sascorer.calculateScore(mol)
    qed = round(QED.weights_max(mol), 2)
    mw = Descriptors.ExactMolWt(mol)
    logp = round(Descriptors.MolLogP(mol), 2)
    tpsa = round(Descriptors.TPSA(mol), 2)
    nhd = Lipinski.NumHDonors(mol)
    nha = Lipinski.NumHAcceptors(mol)

    return smi, sa, qed, mw, logp, tpsa, nhd, nha

def apply_thresholds(df_gen_props, property_thresholds):
    '''
    '''

    df_gen_props_t = df_gen_props[df_gen_props['QED'] >= property_thresholds[0]]
    print('  - Compounds after QED:', len(df_gen_props_t))
    df_gen_props_t = df_gen_props_t[df_gen_props_t['SAscore'] <= property_thresholds[1]]
    print('  - Compounds after SAscore:', len(df_gen_props_t))
    df_gen_props_t = df_gen_props_t[df_gen_props_t['tan_max'] <= property_thresholds[2]]
    print('  - Compounds after tanimoto:', len(df_gen_props_t))

    df_gen_props_t = df_gen_props_t[['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA', 'tan_mean', 'tan_max']]
    return df_gen_props_t

def histogram_properties(df_initial, df_generated, property, out_dir):
    '''
    '''

    plt.figure(figsize=(15,10))
    sns.histplot(data=df_initial, x=property, color='gray', kde=True)
    sns.histplot(data=df_generated, x=property, color='lightgreen', kde=True)

    plt.title('{} Distributions'.format(property))
    plt.suptitle('Run: {}'.format(out_dir.split('/')[-1]))
    plt.legend(['Initial', 'Generated'])

    plt.savefig('{}/{}_{}_hist.png'.format(out_dir, out_dir.format('/')[-1], property), format='png')

def smiles2selfies(smiles_file):
    '''
    '''

    selfies_set = {}

    with open(smiles_file, 'r') as f:
        for smile in f:
            selfies_set[smile.strip()] = sf.encoder(smile.strip())

    char_alphabet =  sf.get_alphabet_from_selfies(selfies_set.values())
    char_alphabet.add("[nop]")
    char_alphabet = list(sorted(char_alphabet))

    return selfies_set, char_alphabet

def selfies_one_hot_encoder(selfies_set, char_alphabet):
    '''
    '''
    discard = 0

    one_hot_selfies = []
    labels = []

    pad_to_len = max(sf.len_selfies(s) for s in selfies_set)  # 5
    symbol_to_idx = {s: i for i, s in enumerate(char_alphabet)}

    for selfie in selfies_set:
        try:
            label, one_hot = sf.selfies_to_encoding(
            selfies=selfie,
            vocab_stoi=symbol_to_idx,
            pad_to_len=pad_to_len,
            enc_type="both"
            )

            one_hot_selfies.append(one_hot)
            labels.append(label)
        except:
            discard += 1

    print(discard)

    return np.array(one_hot_selfies)[:,0:-1,:], labels, np.array(one_hot_selfies)[:,1:,:]

def selfies_one_hot_decoder(one_hot_selfies, alphabet_char):
    '''
    '''

    selfies = []

    for one_hot_encoding in one_hot_selfies:
        selfie = sf.encoding_to_selfies(encoding=one_hot_encoding,
        vocab_itos=alphabet_char,
        enc_type='both'
        )

        selfies.append(selfie)

    return selfies

def remove_duplicates_tanimoto(set1, set2, threshold=1):
    """Remove duplicates molecules from set2 based on tanimoto similarity to
       to molecules in set1."""

    fingerprints_set1 = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in set1]

    # Compute similarity and apply the threshold
    unique_molecules_set2 = []
    for mol in set2:
        fp_mol = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        similarities = DataStructs.BulkTanimotoSimilarity(fp_mol, fingerprints_set1)

        if not any(sim >= threshold for sim in similarities):
            unique_molecules_set2.append(mol)

    return unique_molecules_set2

def filter_unique_molecules(df1, df2, threshold=1):
    """Remove duplicates from the second dataframe it they already existed in
       the first df. It is applied when merging dfs between inners."""

    set1 = [Chem.MolFromSmiles(smi) for smi in df1['smiles']]
    set2 = [Chem.MolFromSmiles(smi) for smi in df2['smiles']]

    # Ensure no NoneType molecules (invalid SMILES)
    set1 = [mol for mol in set1 if mol is not None]
    set2_valid = [(mol, idx) for idx, mol in enumerate(set2) if mol is not None]

    # Apply remove duplicates
    set2_mols, set2_indices = zip(*set2_valid) if set2_valid else ([], [])
    unique_molecules_set2 = remove_duplicates_tanimoto(set1, set2_mols, threshold=threshold)

    # Get the indices of unique molecules and filter the second dataframe
    unique_indices = [set2_indices[i] for i, mol in enumerate(set2_mols) if mol in unique_molecules_set2]
    filtered_df2 = df2.iloc[unique_indices].reset_index(drop=True)

    return df1, filtered_df2

