import pdb
from re import L
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.model_selection import train_test_split

from data.data_processing import length_scanner, load_data, one_hot_encoder, \
    Data_Generator, int_to_smiles, seq_cleaning, smiles_to_int, smiles_cleaning, \
    amino_to_int, protein_one_hot_encoder
from models.generative_models import lstm_model, generate
from utils.utils import apply_thresholds, selfies_one_hot_encoder, \
    smiles2selfies, write_report, plot_acc_loss, compute_properties, histogram_properties, tanimoto2, \
    compute_tanimoto_init, filter_unique_molecules

import random
import argparse as ap
import pandas as pd
import glob

from tqdm import tqdm

import os
import time
import rdkit.Chem as Chem

def parseArg():
    '''
    Parse all the parameters
    '''

    parser = ap.ArgumentParser()

    parser.add_argument("-g", "--general_set_smiles", required=True, type=str, help="path to smiles file.")
    parser.add_argument("-e", "--epochs", required=True, type=int, default=100, help="number of epochs.")
    parser.add_argument("-v", "--validation_size", required=True, type=float, default=0.1, help="validation set size.")
    parser.add_argument("-t", "--test_size", required=True, type=float, default=0.1, help="test set size.")
    parser.add_argument("-b", "--batch_size", required=True, type=int, default=100, help="batch size.")
    parser.add_argument("-s", "--sampling_temperature", required=False, type=float, default=1.2, help="sampling temperature.")
    parser.add_argument("-q", "--quantity", required=True, type=int, default=1000, help="amount of molecules to be generated.")
    parser.add_argument("-n", "--name", required=True, type=str, default="algen_run", help="name of generative run.")
    parser.add_argument("-o", "--outdir", required=True, type=str, default="results", help="output directory.")
    parser.add_argument("-pa", "--specific_set_smiles", required=False, type=str, help="path to alternative smiles file.")
    parser.add_argument("-pt", "--pretrained", required=False, type=str, help="restart training from stored weights. Path to weights.")
    parser.add_argument("-ial", "--inner_al", required=False, type=int, default=10, help="number of active learning (AL) iterations.")
    parser.add_argument("-qed", "--druglikeness", required=False, type=float, default=0.6, help="threshold for QED value of generated molecules.")
    parser.add_argument("-sa", "--sascore", required=False, type=float, default=6, help="threshold for SAscore value of generated molecules.")
    parser.add_argument("-ta", "--tanimoto", required=False, type=float, default=0.6, help="threshold for tanimoto score of generated molecules.")
    parser.add_argument("-r", "--restart", required=False, action="store_true", help="restart job from last stop.")

    args = parser.parse_args()

    # If inner active learning, thresholds for druglikeness, sascore, and tanimoto are required
    if args.inner_al and not all([args.druglikeness, args.sascore, args.tanimoto]):
        parser.error("--inner_al requires --druglikeness, --sascore, and --tanimoto.")

    # Choose or restart or pretrained, but not both
    if args.restart and args.pretrained:
        parser.error("--restart and --pretrained can't run simultaneously.")

    # Define variables
    general_smiles = args.general_set_smiles
    num_epochs = args.epochs
    val_size = args.validation_size
    test_size = args.test_size
    batch_size = args.batch_size
    quantity = args.quantity
    name = args.name
    outdir = args.outdir
    specific_smiles = args.specific_set_smiles
    weights = args.pretrained
    qed_t = args.druglikeness
    sascore_t = args.sascore
    tanimoto_t = args.tanimoto
    restart = args.restart

    # Define default variables for not required arguments
    if args.sampling_temperature:
        sampling_temperature = args.sampling_temperature
    else:
        # If we do not provide a sampling temperature, generate a random one from 0.1 to 1.5
        sampling_temperature = random.uniform(0.1, 1.5)

    if args.inner_al:
        inner_al = args.inner_al
    else:
        inner_al = 1

    return general_smiles, num_epochs, val_size, test_size, batch_size, sampling_temperature, quantity,\
    name, outdir, specific_smiles, weights, inner_al, qed_t, sascore_t, tanimoto_t, restart


def split_data(dat, validation_size, test_size, maximum_len, char_alph=None):
    '''
    Args:
    - dat: The input data to be split
    - validation_size: The proportion of the data to be used for validation
    - test_size: The proportion of the data to be used for testing
    - maximum_len: Max length
    - char_alph: List of the unique characters found in a SMILE.

    Operations:
    - Split the dataset into training, validation and test sets.
    - One hot encoding of the predictors (x) and target (y)

    Return:
    - All sets
    '''

    # Split the dataset into train, test and validation
    pre_train, validation = train_test_split(dat, test_size=validation_size, random_state=42)
    train, test = train_test_split(pre_train, test_size=test_size, random_state=42)

    print('train: ' , type(train))
    print("\nTraining set size: ", train.shape[0])
    print("Validation set size: ", validation.shape[0])
    print("Test set size: ", test.shape[0])
    print("Number of possible characters: ", len(int_to_smiles))

    # One-hot-encoding. The resulting object is a 3D Tensor (Number of SMILES x Length of SMILES x Number of characters in a SMILE)
    train_pred, train_target = one_hot_encoder(smiles=train, max_len=maximum_len, num_char=len(smiles_to_int))
    val_pred, val_target = one_hot_encoder(smiles=validation,  max_len=maximum_len, num_char=len(smiles_to_int))
    test_pred, test_target = one_hot_encoder(smiles=test,  max_len=maximum_len, num_char=len(smiles_to_int))

    return train, validation, test, train_pred, train_target, val_pred, val_target, test_pred, test_target


def train_and_evaluate_model(model, train_pred, train_target, val_pred, val_target, test_pred, test_target, b_size, n_epochs, output_directory, name, checkpoint_name, moment, deb):
    '''
    Args:
    - model: The model to be trained and evaluated.
    - train_pred: The input data for training.
    - train_target: The ground truth labels for training.
    - val_pred: The input data for validation.
    - val_target: The ground truth labels for validation.
    - test_pred: The input data for testing.
    - test_target: The ground truth labels for testing.
    - b_size: The batch size used during training.
    - n_epochs: The number of epochs for training.
    - output_directory: The directory to save checkpoints and plots.
    - name: The name prefix for saved files.
    - checkpoint_name: The name of the checkpoint file.
    - moment: The moment of the training (Start, Transfer, Restart)
    - deb: Debugging mode (run_eagerly)

    Operations:
    - Train the model using the given generators, evaluate its performance and save relevant information (checkpoint)

    Returns:
    - score: The evaluation loss on the test set.
    - acc: The evaluation accuracy on the test set.
    '''

    # Number of steps in each epoch
    steps_per_epoch = len(train_pred) // b_size

    # Generate a learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.0001, decay_steps=steps_per_epoch*50, decay_rate=1.0, staircase=False)
    opt = Adam()

    # Configure the model for training
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'], run_eagerly=False)

    training_generator = Data_Generator(train_pred, train_target, b_size)
    validation_generator = Data_Generator(val_pred, val_target, b_size)

    # Define checkpoint
    if moment == 'Start':
        checkpoint = ModelCheckpoint(output_directory + '/' + name + checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto', save_freq='epoch')
        callbacks_list = [checkpoint]

    elif moment == 'Transfer':
        print('Model checkpoint: ', output_directory + '/' + name + checkpoint_name)
        checkpoint = ModelCheckpoint(output_directory + '/' + name + checkpoint_name, monitor='val_loss', verbose = 1, save_weights_only = True, mode ='auto', save_freq='epoch')
        callback =  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
        callbacks_list = [callback]

    elif moment == 'Restart':
        checkpoint = ModelCheckpoint(output_directory + '/' + name + checkpoint_name, monitor='val_loss', verbose = 1, save_weights_only = True, mode ='auto')
        callbacks_list = [checkpoint]

    # Train the model
    validation_steps = len(val_pred) // b_size
    history = model.fit(training_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs, verbose=1, validation_data=validation_generator, validation_steps=validation_steps,
                        shuffle=True, callbacks=callbacks_list)

    # Generate the accuracy and loss plots
    plot_acc_loss(history=history, name=name, output_dir=output_directory)

    # Evaluate the model
    score, acc = model.evaluate([test_pred, test_pred], test_target, batch_size=b_size, verbose=0)

    return score, acc


def computing_properties_spec_set(dat, output_dir, name):
    '''
    Args:
    - dat: Specific set with the compounds
    - output_dir: Name of the output directory
    - name: Base name of the CSV file generated

    Operations:
    - Compute the properties of the compounds of the specific set
    - Save them in a dataframe

    Returns:
    - Dataframe containing the properties of each compound of the specific set
    '''

    data_init = []
    print('Computing properties for specific set...')

    # Compute the properties of each compound
    for comp in tqdm(dat.tolist()):
        try:
            smi, sa, qed, mw, logp, tpsa, nhd, nha = compute_properties(comp)
            data_init.append([smi, sa, qed, mw, logp, tpsa, nhd, nha])
        except:
            pass # if not valid compound

    # Save the properties in a dataframe
    df_init_props = pd.DataFrame(data_init, columns=['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA'])
    df_init_props.to_csv(output_dir + '/' + name + '_specific_smiles.csv')

    return df_init_props


def computing_properties_generated_set(smiles, t_smiles, output_dir, name, df_init_props, mom):
    '''
    Args:
    - smiles: Generated compounds + specific training set
    - t_smiles:
    - output_dir: Name of the output directory
    - name: Base name of the CSV file generated
    - df_init_props: Properties of the specific set
    - mom: The moment of the training (Start, Transfer, Restart)

    Operations:
    - Compute properties of the compounds of the generated compounds
    - Generate histograms for the comparison of properties (specific vs generated)
    - Save them in a dataframe

    Return:
    - Dataframe containing the properties of each compounds of the general set
    '''

    data = []
    print('Computing properties...')

    if mom == 'restart':
        simles_fps = smiles

    elif mom == 'initial':
        smiles_fps = compute_tanimoto_init(smiles)

    # Compute properties of the general set
    for comp in tqdm(t_smiles):
        try:
            smi, sa, qed, mw, logp, tpsa, nhd, nha = compute_properties(comp)

            if mom == 'initial' or 'restart':
                tan_mean, tan_max = tanimoto2(comp, smiles_fps)
                data.append([smi, sa, qed, mw, logp, tpsa, nhd, nha, tan_mean, tan_max])

            else:
                data.append([smi, sa, qed, mw, logp, tpsa, nhd, nha])

        except:
            pass

    # Save the properties in a dataframe
    if mom == 'initial' or 'restart':
        df_gen_props = pd.DataFrame(data, columns=['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA', 'tan_mean', 'tan_max'])

        # Generate histograms
        for prop in list(df_gen_props.columns)[1:-3]:
            histogram_properties(df_init_props, df_gen_props, prop, output_dir)

    else:
        df_gen_props = pd.DataFrame(data, columns=['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA'])


    # Save the dataframe
    df_gen_props_simple = df_gen_props[['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA', 'tan_mean', 'tan_max']]
    df_gen_props_simple.to_csv(output_dir + '/' + name + '_generated_smiles.csv')
    print("\nSmiles written to " + output_dir + '/' + name + '_generated_smiles.csv')

    return df_gen_props

def model_vae(model, size, test_pred):
    '''
    Args:
        model (tf.keras.Model): The VAE model from which components will be extracted.
        size (int): The size of the vocabulary.
        test_pred (numpy.ndarray): The test data used to predict the latent space representations.

    Operations:
        1. Extract the encoder part of the VAE model.
        2. Create a model to map latent space vectors to LSTM states.
        3. Create a decoder model for sequence generation.
        4. Copy weights from the original VAE model to the decoder model.
        5. Predict the latent space representation for the test data.
        6. Randomly select a latent space vector as the seed for sequence generation.
        7. Set a scale factor for sequence generation.

    Returns:
        gen_model (tf.keras.Model): The decoder model for sequence generation.
        latent_to_states_model (tf.keras.Model): The model mapping latent space vectors to LSTM states.
        latent_seed (numpy.ndarray): The seed for generating sequences.
        scale (float): The scale factor for sequence generation.
    '''

    encoder = Model(inputs=model.layers[0].input, outputs=model.layers[3].output)

    latent_input = Input(shape=(128, ))
    state_h = model.layers[5](latent_input)
    state_c = model.layers[6](latent_input)
    latent_to_states_model = Model(latent_input, [state_h, state_c])
    print('\n')
    latent_to_states_model.summary()

    decoder_inputs = Input(batch_shape=(1, 1, size))
    decoder_lstm = LSTM(256, return_sequences=True, stateful=True)(decoder_inputs)
    decoder_outputs = Dense(size, activation='softmax')(decoder_lstm)
    gen_model = Model(decoder_inputs, decoder_outputs)

    for i in range(1,3):
        gen_model.layers[i].set_weights(model.layers[i+6].get_weights())

    print('\n')
    gen_model.summary()

    test_latent_space = encoder.predict(test_pred)

    rand_seed = random.randint(0, test_latent_space.shape[0])

    latent_seed = test_latent_space[rand_seed:rand_seed + 1]

    scale = 0.5

    return gen_model, latent_to_states_model, latent_seed, scale

def main():

    time0 = time.time()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print("Error setting GPU config:", e)

    # Set all variables to the given argument values
    general_smiles, num_epochs, val_size, test_size, batch_size, sampling_temp, quant, name, outdir,\
    specific_smiles, weights_file, inner_al, qed_t, sascore_t, tanimoto_t, restart = parseArg()

    # Print relevant information to log
    print('PARAMETERS:')
    print('  - General training set:', general_smiles)
    print('  - Specific training set:', specific_smiles)
    print('  - Weights file from pretrained:', weights_file)
    print('  - Name:', name)
    print('  - Output directory:', outdir)
    print('  - Num epochs:', num_epochs)
    print('  - Validation size:', val_size)
    print('  - Test size:', test_size)
    print('  - Batch size:', batch_size)
    print('  - Sampling Temperature:', sampling_temp)
    print('  - Quantity:', quant)
    print('  - Inner Active Learning Cycles:', inner_al)
    print('  - QED threshold:', qed_t)
    print('  - SAscore threshold:', sascore_t)
    print('  - Tanimoto threshold:', tanimoto_t)
    print('\n')

    # Generate the output directory
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    inner_output_dir = outdir + '/' + name

    if restart:
        print('Restarting the job')
        # Define paths according to the rounds that have been done previously
        round = glob.glob('%s_*'%inner_output_dir)
        round = [os.path.isdir(x) for x in round]
        round = len(round)
        print(f'\nRounds done: {round-1}\n')
    else:
        round = 1
        if not os.path.exists(inner_output_dir + '_' + str(round)):
            os.mkdir(inner_output_dir + '_' + str(round))

    # Generate a plot with the length of the SMILES of the general training set and return 2*median of the length of the SMILES
    max_len = length_scanner(general_smiles)

    # Transform each of the SMILES from the general training set into an array. The ones that have a length higher than max_len will be removed
    data_general = load_data(data=general_smiles, max_len=max_len)

    # Clean SMILES from salts and charges and standardize them
    data_general = smiles_cleaning(data=data_general)

    # Number of characters in SMILES (==50)
    n_vocab = len(smiles_to_int)

    # Split the general training dataset into train, test and validation one-hot-encoded
    train, validation, test, train_x, train_y, val_x, val_y, test_x, test_y = split_data(data_general, val_size, test_size, max_len)

    # Define the LSTM VAE model architecture based on the input and output dimensions of the data
    model = lstm_model(train_x, train_y)
    model.summary()

    # If pretrained weights file is provided, load these weights. If not, run the
    # training of the general training set and then load the resulting weights.
    if not weights_file and not restart:
        print('Training of the general set...')
        training_dir = outdir + '/pretrained'
        if not os.path.exists(training_dir):
            os.mkdir(training_dir)

        train_and_evaluate_model(model, train_x, train_y, val_x, val_y, test_x, test_y, batch_size, num_epochs, training_dir, 'general', '_weights.hdf5', 'Start', True)
        weights_file = training_dir + '/general_weights.hdf5'

    initial_specific = specific_smiles

    timegt = time.time() - time0
    print('Time training general: ', timegt, 'seconds.', 'Human readable:', time.strftime("%H:%M:%S", time.gmtime(timegt)))

    # For each inner cycle of active learning:
    for i in range(round, inner_al + 1):
        timeal0 = time.time()

        if restart:
            # define specific set from the previous round
            specific_smiles = inner_output_dir + '_' + str(i) + '/' + name + '_generated_smiles_merged.csv'
            print('Specific set: ', specific_smiles)

        # Transform each of the SMILES from the specific training set into an array. The ones that have a length higher than max_len will be removed
        data_specific = load_data(data=specific_smiles, max_len=max_len)
        print('Load_data: ', data_specific)
        data_specific = smiles_cleaning(data=data_specific)
        print('Smiles_cleaning: ', len(data_specific))

        # Compute properties of the specific training set
        df_init_props = computing_properties_spec_set(data_specific, inner_output_dir+'_'+str(i), name)

        # Split the specific training set into train, test and validation one-hot-encoded
        train, validation, test, train_x, train_y, val_x, val_y, test_x, test_y = split_data(data_specific, val_size, test_size, max_len)
        n_vocab = len(smiles_to_int)

        print(f'\nROUND {i}')

        # Load weights
        if round == 1:
            print('loading weights for first round ...')
            model.load_weights(weights_file)
        else:
            print('loading weights for round %s...'%round)
            model.load_weights(inner_output_dir + '_1' + '/' + name + '_weights.hdf5')

        # Batch size is all the specific training set
        batch_size = len(val_x) - 2    # Specific batch size not given by parser!
        num_epochs = 100               # Specific num epochs not given by parser!

        # Train, evaluate the model and generate checkpoints using the specific training set
        score, acc = train_and_evaluate_model(model, train_x, train_y, val_x, val_y, test_x, test_y, batch_size, num_epochs, inner_output_dir+'_'+str(i), name, '_weights.hdf5', 'Transfer', True)

        print(f'\nTest alternative loss: {score}')
        print(f'Test alternative accuracy: {acc}')

        model.save_weights(inner_output_dir+'_'+str(i) + '/' + name + '_weights.hdf5')

        gen_model, latent_to_states_model, latent_seed, scale = model_vae(model, len(int_to_smiles), test_x)

        # Evaluate the model
        score, acc = model.evaluate([test_x, test_x], test_y, batch_size=batch_size, verbose=0)
        print(f'\nTest loss: {score}')
        print(f'Test accuracy: {acc}')

        timest = time.time() - timeal0
        print('Time training specific: ', timest, 'seconds.', 'Human readable:', time.strftime("%H:%M:%S", time.gmtime(timest)))

        # Generate new compounds
        timeg0 = time.time()
        t_mols, t_smiles, not_valid = generate(gen_model, latent_to_states_model, latent_seed, sampling_temp, n_vocab, scale, quant)

        print('GENERATED COMPOUNDS')
        print(len(t_smiles))

        timeg = time.time() - timeg0
        print('Time generation: ', timeg, 'seconds.', 'Human readable:', time.strftime("%H:%M:%S", time.gmtime(timeg)))

        if len(t_mols)<=0:
            raise Exception("No valid compounds generated.")

        write_report(train_size=train.shape[0], val_size=validation.shape[0], test_size=test.shape[0], test_loss=score,
                    test_acc=acc, epochs=num_epochs, batch_size=batch_size, sampling_temp=sampling_temp, quant=quant,
                    not_valid=not_valid, name=name, outdir=inner_output_dir+'_'+str(i))

        timef0 = time.time()

        # Compute properties of the generated compounds
        df_gen_props = computing_properties_generated_set(specific_smiles, t_smiles, inner_output_dir+'_'+str(i), name, df_init_props, 'initial')

        # Apply Thresholds (QED, SA and Tanimoto)
        df_filtered = apply_thresholds(df_gen_props=df_gen_props, property_thresholds=[qed_t, sascore_t, tanimoto_t])
        print('AFTER FILTER:')
        print(len(df_filtered))

        if len(df_filtered)==0:
            raise Exception("No valid compounds generated after the last filter.")

        timef = time.time() - timef0
        print('Time Filter thresholds Inner AL: ', timef, 'seconds.', 'Human readable:', time.strftime("%H:%M:%S", time.gmtime(timef)))

        if not os.path.exists(inner_output_dir + '_' + str(i+1)):
            os.mkdir(inner_output_dir + '_' + str(i+1))

        # Generate a CSV with the filtered compounds
        df_filtered.to_csv(inner_output_dir + '_' + str(i+1) + '/' + name + '_generated_smiles_threshold_' + str(i) + '.csv')

        # Merge the generated compounds with the compounds from the specific training set
        df_init_props, df_filtered = filter_unique_molecules(df_init_props, df_filtered) # remove duplicates
        df_merge_rounds = pd.concat([df_filtered['smiles'], df_init_props['smiles']])
        df_merge_rounds.reset_index(inplace=True, drop=True)

        df_merge_rounds.to_csv(inner_output_dir + '_' + str(i+1) + '/' + name + '_generated_smiles_merged.csv', index=False, header=None)

        # Change the name of the specific training set to the new set (generated + previous specific training set)
        specific_smiles = inner_output_dir + '_' + str(i+1) + '/' + name + '_generated_smiles_merged.csv'

        timeal = time.time() - timeal0
        print('Time Inner AL cycle: ', timeal, 'seconds.', 'Human readable:', time.strftime("%H:%M:%S", time.gmtime(timeal)))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Shutting down with error: ", e)
