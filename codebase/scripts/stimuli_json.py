import os
import numpy as np
import json
import re
import pandas as pd
from nltk.corpus import wordnet as wn

def select_clean_sentences(sentences, indices, limit = 50):
    """
    sentences- list of sentences in SEMCOR for a type
    indices- list of indices for the sense
    limit- maximum length of sentence (in words)
    
    Returns a randomly selected sentence from SEMCOR 
    """
    short_sentences = [sentences[i] for i in indices if len(sentences[i].split(" ")) <= limit]
    selected = np.random.choice(short_sentences)
    selected = re.sub(r"\s([?.!,`'](?:\s|$))", r'\1', selected) #Clean out special characters
    return selected
    
sense_as_str = lambda s: s.split('(')[1].replace(')', '')


semcor_lemmas = pd.read_csv('../data/expt_semcor_types.csv')["Lemma"].tolist() #SEMCOR words based on entropy
with open('../data/polysemy_words.txt', 'r') as f: #Types from S&R and types for shared trials
    picked_lemmas = [l.strip() for l in f.readlines()]
lemmas = semcor_lemmas + picked_lemmas
make_fname = lambda x: x.split('.')[0] + '_' + x.split('.')[1] + '.json'
lemmas = [make_fname(l) for l in lemmas] 
datapath = os.path.join('..', 'data', 'pipeline_results', 'semcor') #Location of the type's SEMCOR sentences
inputs = {}

for l in lemmas:
    with open(os.path.join(datapath, l), 'r') as f:
        logged_data = json.load(f)    
    assert len(logged_data['sense_indices']) == len(logged_data['sense_names'])
    lem = logged_data['lemma']
    lem = lem.replace('.', '_') #for FB
    inputs[lem] = {}
    for s in logged_data['sense_names']:
        l = sense_as_str(s).strip()
        #Needed to change NLTK source code for this to work
        #strip single quotes in the synset function in nltk.corpus.reader.wordnets
        pooja_synset = wn.synset(l)
        defn = pooja_synset.definition()
        l = pooja_synset.name()
        l = l.replace('.', '_')
        inputs[lem][l] = {}
        inputs[lem][l]['def'] = defn
        indices_for_sense = np.argwhere(np.array(logged_data['sense_labels']) == pooja_synset.name()).flatten()
        #Gets indices in the list of sentences where the sense is used
        random_sentence = select_clean_sentences(logged_data['original_sentences'], indices_for_sense)
        inputs[lem][l]["sent"] = random_sentence
    inputs[lem]['senses'] = len(logged_data['sense_names'])
    inputs[lem]['type'] = lem

final_dict = {'polysemy_pilesort': {
    'inputs': inputs
}}
with open(os.path.join("..", "data", "stimuli.json"), 'w') as f:
    json.dump(final_dict, f)