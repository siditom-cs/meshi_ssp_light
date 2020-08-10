# -*- coding: utf-8 -*-
import sys
import re
import numpy as np
import pandas as pd
from os import listdir
from os.path import isdir, join
from preprocess import *



def sort_columns(decoder, pred):
    """Sorts the output prediction from model z (with decoder 'decoder') to align with the default decoder 'dec'.
        Needed, as different models might have different decoders.
    
    Args:
        decoder - the model's decoder
        pred - the prediction matrix with shape (seq_max_len = 700, n_classes=14)
    
    Returns:
        Sorted prediction sequences with classes ordered as the default decoder 'dec'.
    """
    global dec
    items = {key: value for key, value in decoder.word_index.items()}
    indices = list()
    for i in range(1,len(items)+1):
        indices.append(items[dec[i]])
    return pred[:,:,indices]

def get_model_pred(mpath,raw_seq,prof,iModel):
    """Sequence class prediction of (raw_seq,prof) with model in mpath and model type iModel.
    
    Args:
        mpath - model directory path
        raw_seq,prof - the protein's data (fasta and PSSM profile)
        iModel - model type, needed for specific preprocessing procedures.
    
    Returns:
        Prediction sequnece of size (700,n_class) by model in mpath
    """
    maxlen_seq = 700
    model, decoder, encoder = load(mpath)
    if iModel == 2:
        input_data, input_data_alt, prof_np = preprocess_data(raw_seq, prof, encoder, decoder, 768)
    else:
        input_data, input_data_alt, prof_np = preprocess_data(raw_seq, prof, encoder, decoder, maxlen_seq )
    inp = [input_data_alt, prof_np]
    if iModel == 1:
        inp = [input_data, input_data_alt, prof_np]
    if iModel == 4:
        alpha = 0.5
        input_fofe = np.array(list(map(lambda x:encode_FOFE(x, alpha, maxlen_seq), input_data)))
        input_data_ = np.concatenate([input_data, prof_np], axis=2)
        inp = np.concatenate((input_data_,input_fofe), axis=2)

    pred = sort_columns(decoder, model.predict(inp))
    pred = pred[:,0:maxlen_seq,:]
    return  pred
    
def get_allpreds(raw_seq,prof):
    """Get sequence predictions of (raw_seq,prof) as predicted by each of the models in ./models/ directory.
    
    Returns:
        3D numpy array of size (#models,700,n_classes)
        
    Assumptions:
        The models directories names follows a specific rule. The model type is given by the [-7] letter of the name.
    """
    models_path = APP_PATH + "/models/" + Model_type + "/"
    models_dirs = [join(models_path, f) for f in listdir(models_path) if isdir(join(models_path, f))]
    stats = None
    for mpath in models_dirs:
        iModel = int(mpath[-7])
        st = get_model_pred(mpath,raw_seq,prof, iModel)
        if stats is None:
            stats = st
        else:    
            stats = np.vstack((stats,st))
    return stats

def get_avg_pred(stats):
    """Procude an ensemble prediction for each residue in the sequence by averaging the predictions from the given models.
    
    Returns:
        2D numpy array of size (700,n_classes). A class distribution for each residue.
    """
    avg_stats = np.average(stats,axis=0)
    return avg_stats

def get_partial_preds(rseq_lst, prof_lst, inx):
    """Produce the average ensemble prediction for each partial sequence of the protein.
    
    Args:
        rseq_lst,prof_lst,inx - list of partial protein feature matrices with length <700 each.
    
    Returns:
        List of partial prediction by the average ensemble model.
        
    """
    stats_lst = []
    for i in range(0,len(rseq_lst)):
        stats = get_allpreds(rseq_lst[i],prof_lst[i])
        stats = get_avg_pred(stats)
        stats_lst.append(stats)
    return stats_lst

def combine_arrayed_predictions(stats_lst):
    """Combines the partial average ensemble predictions with length < 700 into a single prediction with the original protein's length.
    
    Args:
        stats_lst - list of 2D matrices, each represents the per-residue sequence predictions of a partial sequence.
    
    Returns:
        2D matrix of per-residue class distributions of size (sequence_length, n_classes).
        
    Note: 
        Each partial prediction overlaps with its next partial prediction with 100 residues, the results prediction averages these overlapping residues predictions.
    """
    shp = stats_lst[0].shape
    stats = np.zeros([len(stats_lst)*shp[0],shp[1]],dtype=float)
    ilres = shp[0]
    nclass = shp[1]
    stats[0:ilres,0:nclass] = stats_lst[0]
    for i in range(1, len(stats_lst)):
        prev100stats = stats[ilres-100:ilres,0:nclass] 
        next100stats = stats_lst[i][0:100,0:nclass]
        overlap_stats = np.vstack((prev100stats.reshape([1,100,shp[1]]), next100stats.reshape([1,100,shp[1]]) ))
        avg_overlap_stats = np.average(overlap_stats, axis=0)
        stats[ ilres-100:ilres, 0:nclass] = avg_overlap_stats 
        stats[ ilres:ilres+shp[0]-100, 0:nclass] = stats_lst[i][100:shp[0],0:nclass]
        ilres = ilres + shp[0]
    return stats