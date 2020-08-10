# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model, Input, Sequential, model_from_json, load_model
import pickle

APP_PATH = "C:\\Users\\Maayan\\Desktop\\siditom\\Khealth\\meshi_ssp_light2" #"/home/cluster/users/siditom/data/phd/meshi_ssp"
Model_type = "rw13" #options: rw13 | rw8 | nr8 | nr13
dec13 = {1: 'h', 2: 'c', 3: 't', 4: 'z', 5: 's', 6: '-', 7: 'a', 8: 'g', 9: 'q', 10: 'p', 11: 'e', 12: 'b', 13: 'm', 14: 'i'}
dec8 = {1: 'h', 2: 'e', 3: 'c', 4: 't', 5: 's', 6: '-', 7: 'g', 8: 'b', 9: 'i'}
dec = dec13

def load(path):
    """Loads a pre-trained model to memory.
    
    Args:
        The path to the directory with the model's weights and parameters.
        
    Returns:
        A keras DL model, its data decoder, and encoder.
    """
    json_file = open(path+'/model_e150.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+"/model_e150.h5")
    model = loaded_model
    # loading encoder and decoder
    with open(path+'/encoder.pickle', 'rb') as handle:
        encoder = pickle.load(handle)
    with open(path+'/decoder.pickle', 'rb') as handle:
        decoder = pickle.load(handle)
    return model, decoder, encoder

def seq2onehot(seq, n, maxlen_seq = 700):
    """Transforn a protein fasta seqeunce of letters to one-hot matrix with n-classes.
    """
    out = np.zeros((len(seq), maxlen_seq, n))
    for i in range(len(seq)):
        for j in range(maxlen_seq):
            out[i, j, seq[i, j]] = 1
    return out

def seq2ngrams(seqs, n = 1):
    """Computes and returns the n-grams of a particualr sequence, defaults to trigrams
    """
    return [[seq[i : i + n] for i in range(len(seq))] for seq in seqs]

def preprocess_data(raw_seq, prof, encoder, decoder, maxlen_seq):
    """Appends the feature matrices of the proteins according to the model's encoder and decoder.
    
    Args:
        raw_seq - the sequence fasta string
        prof - the protein's PSSM file from psi-blast
        encoder - the model's encoder
        decoder - the model's decoder
        maxlen_seq - the model's predefined maximal sequence length. default = 700.
        
    Returns:
        The feature matrix representing the protein.
    """
    ngrams = 1
    input_grams = seq2ngrams([raw_seq],ngrams)

    # Initializing and defining the tokenizer encoders and decoders based on the train set
    tokenizer_encoder = encoder
    tokenizer_decoder = decoder

    # Inputs
    input_data = tokenizer_encoder.texts_to_sequences(input_grams)
    input_data = sequence.pad_sequences(input_data, maxlen = maxlen_seq, padding = 'post')

    # Computing the number of words and number of tags to be passed as parameters to the keras model
    n_words = len(tokenizer_encoder.word_index) + 1
    n_tags = len(tokenizer_decoder.word_index) + 1

    input_data_alt = input_data
    input_data = seq2onehot(input_data, n_words)

    prof_np = np.append(prof, np.zeros([1,(maxlen_seq - int(len(raw_seq)))*22],dtype=float))
    prof_np = prof_np.reshape((1, maxlen_seq, 22))
    return input_data, input_data_alt, prof_np

# 
def encode_FOFE(onehot, alpha, maxlen):
    """Fixed-size Ordinally Forgetting Encoding. Model4 data preprocessing prequsities.
    """
    enc = np.zeros((maxlen, 2 * 22))
    enc[0, :22] = onehot[0]
    enc[maxlen-1, 22:] = onehot[maxlen-1]
    for i in range(1, maxlen):
        enc[i, :22] = enc[i-1, :22] * alpha + onehot[i]
        enc[maxlen-i-1, 22:] = enc[maxlen-i, 22:] * alpha + onehot[maxlen-i-1]
    return enc


def get_data(fasta_path, profile_path):
    """upload to memory the protein's data.
    
    Args:
        fasta_path - the path to the fasta file of the protein.
        profile_path - the path to the pre-calculated by psi-blast PSSM matrix.
        
    Returns:
        A string (fasta sequence), and a flattened profile matrix (22*length_of_sequence)
    
    """
    with open(fasta_path,"r") as f:
        pid = f.readline()[1:]
        raw_seq = f.readline().replace('\n','')
    with open(profile_path,"r") as f:
        lines = f.read().split("\n")
        prof = [re.split(r'\s+',l)[3:25] for l in lines]
        prof_seq = [re.split(r'\s+',l)[2:3] for l in lines]
        prof = prof[3:-7]
        prof_seq = ''.join([c[0] for c in prof_seq[3:-7]])
    prof = str(prof).replace("][",",").replace("[","").replace("]","").replace("."," ").replace(","," ").replace("'"," ")
    prof = re.sub(' +',' ',prof).strip()    
    assert(len(raw_seq) == len(prof_seq))
    return raw_seq, prof

def split_sequence(raw_seq,prof):
    """Splits sequences with length greater than 700 to overlapping sequences with length <= 700.
    
    Args:
        raw_seq - A string (fasta sequence), and a flattened profile matrix (22*length_of_sequence)
        
    Returns: 
        A list of overlapping partial sequences and profiles of (raw_seq,prof)
    """
    inx = []
    rseq_lst = []
    prof_lst = []
    i = 0
    while (i<len(raw_seq)):
        inx.append(i)
        i = i+700
    inx.append(len(raw_seq))
    rseq_lst.append(raw_seq[inx[0]:inx[1]])
    prof_np = np.fromstring(prof, dtype=float,sep=" ")
    prof_lst.append(prof_np[inx[0]:inx[1]*22])
    for i in range(1, len(inx)-1):
        rseq_lst.append(raw_seq[(inx[i]-100):inx[i+1]])
        prof_part = prof_np[(inx[i]-100)*22:inx[i+1]*22]
        prof_lst.append(prof_part)
    return rseq_lst, prof_lst, inx    