import sys
import re
import numpy as np
import pandas as pd
from preprocess import *
from postprocess import *

"""
MESHI-SSP software is a secondary structure prediction of proteins software.

Usage:[App Name] [fasta file] [profile]

Prequsities: single sequence fasta file and a PSSM matrix from psi-blast.

Returns: 13-letter classification of the protein sequence as described in (Sidi & Keasar 2020).


* The software includes pre-trained SSP models with fixed sizeof 700 residues under ./models/rw13/
* It enables the prediction of sequences with arbitrary lengths by splitting and reassembling sequences with more than 700 residues.

"""

def get_args():
    """Get the arguments of the software.
    Note: this version allows a single type of prediction rw13.
    """
    nArgs = len(sys.argv)
    errStr="Usage:[App Name] [fasta file] [profile] [ModelType (rw/nr) - Optional] [nClass (8/13) - Optional]\nDefault values: rw 13"
    if (nArgs-1 > 2) or (nArgs-1 == 3) or (nArgs-1 < 2):
        print(errStr)
        exit()
    fasta_path = sys.argv[1]
    profile_path = sys.argv[2]
    if len(sys.argv)-1 == 4:
        mtype = sys.argv[3]
        nClass = int(sys.argv[4])
        global dec, Model_type
        if ((mtype in {"rw","nr"}) and (nClass in {8,13})):
            Model_type = mtype+str(nClass)
            if (nClass == 8):
                dec = dec8
            elif (nClass == 13):
                dec = dec13
        else:
            print(errStr)
            exit()
    return fasta_path, profile_path 



def save_pred(pred, seq, path = 'out.csv'):
    """Saves the per-residue class-prediction into a csv file.
    """
    headers = [value.upper() for key, value in dec.items()]
    max_inx = np.argmax(pred,axis=1) +1
    max_letters = [dec[int(i)].upper() for i in max_inx]
    pd.options.display.float_format = '${:,.5f}'.format
    df = pd.DataFrame(pred[0:len(seq),:], columns=headers)
    df = df.applymap("{:,.4f}".format)
    df.insert(0,'Seq',list(seq.upper()),True)
    df.insert(1,'Pred', max_letters[0:len(seq)], True)
    df.to_csv(path, index=False, sep='\t')

def main():
    fasta_path,profile_path = get_args()
    out_path = re.sub(r'\.fasta$', '', fasta_path) + "." + Model_type
    raw_seq,prof = get_data(fasta_path,profile_path)
    
    rseq_lst, prof_lst, inx = split_sequence(raw_seq, prof) # Splits the sequence into overlapping partial sequence with length < 700, to accomodate the use of the pretrained models.
    stats_lst = get_partial_preds(rseq_lst, prof_lst, inx) # Predicts the class distributions of each residue in each partial sequence.
    stats = combine_arrayed_predictions(stats_lst) # Combine the partial sequence prediction to a single prediction.
    save_pred(stats, raw_seq, out_path) # Save the prediction to a file.
    
if __name__ == "__main__":
    main()
    

    
