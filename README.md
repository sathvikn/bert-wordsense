# Attention-Based Neural Networks Encode Aspects of Human-Like Word Sense Knowledge

This project evaluated how a contextualized word embedding model (BERT) represents senses of words when compared with human intuition.

`core` contains an interface with BERT and the SEMCOR corpus, code for the logistic regression probe, and other useful functions for analysis.

Results are reported in the `notebooks` directory.

`scripts` has programs to generate experimental stimuli and call the data pipeline. To run it with one word, run `python automation.py --type [word].[pos]`. If you have a list of types in a CSV, it should have the columns "word" and "pos" and its path can be called with `python automation.py --file [filepath to data]`.

Experiment code is at https://github.com/sathvikn/pilesort. 
