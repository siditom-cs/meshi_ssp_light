#########################################################
######################   README  ########################
#########################################################

MeshiRWSSP is a software for 8-alphabet / 13-alphabet secondary structure prediction of proteins.

For a full description of the method please visit (Sidi & Keasar paper).


It is based on Iddo Drori et. al. (https://arxiv.org/abs/1811.07143) DL models.


Execution example:
-----------------

>> python predict.py ./test/6bf4C.fasta ./test/6bf4C.mtx


Installation:
------------

1. For DSSP 13-class alphabet prediction - change the APP_PATH in preprocess.py to the current directory.


Execution: 
---------

[App Name] [fasta file] [profile] [Optional - ]

(*) This version operates only on a single target sequence. 
(*) The sequence length in the fasta file must be the same as the profile length.


Perliminaries:
-------------
(*) python3.4
(*) python packages: Keras2.0, pandas, numpy, TCN, re, pickle,json 
