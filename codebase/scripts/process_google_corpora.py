import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import re
from nltk.corpus import wordnet as wn

def synset_from_sense_key(sense_key):
    sense_key_regex = r"(.*)\%(.*):(.*):(.*):(.*):(.*)"
    synset_types = {1:'n', 2:'v', 3:'a', 4:'r', 5:'s'}
    lemma, ss_type, lex_num, lex_id, head_word, head_id = re.match(sense_key_regex, sense_key).groups()
    ss_idx = '.'.join([lemma, synset_types[int(ss_type)], lex_id])
    return wn.synset(ss_idx)

def parse_xml(fpath, sense_map):
    print("Processing ", fpath)
    contents = ET.parse(fpath)
    root = contents.getroot()
    tags = []
    sentence_indices = []
    for i in range(len(root)):
        child = root[i]
        tags.append(child.attrib)
        if child.attrib['break_level'] == 'SENTENCE_BREAK':
            sentence_indices.append(i)
    
    text_lemmas = []
    for i in range(len(sentence_indices)):
        if i == 0:
            sentence_tokens = tags[:sentence_indices[0]]
        elif i == len(sentence_indices) - 1:
            sentence_tokens = tags[sentence_indices[-1]:]
        else:
            sentence_tokens = tags[sentence_indices[i]:sentence_indices[i + 1]]

        sentence_str = ""
        lemma_sense = [] #List of tuples (lemma, NOAD sense)
        for t in sentence_tokens:
            sentence_str += (t['text'] + " ")
            if 'lemma' in t.keys():
                wn_sense = sense_map[sense_map['NOAD'] == t['sense']]['WordNet'].values
                if len(wn_sense):
                    wn_synsets = []
                    for k in wn_sense[0].split(','):
                        try:
                            wn_synsets.append(synset_from_sense_key(k))
                        except:
                            continue
                    for s in wn_synsets:
                        lemma_sense.append({'lemma': t['lemma'], 'pos': t['pos'], 'noad_sense': t['sense'], 'wn_sense': s, 'source': fpath})

        for l in lemma_sense:
            l['sentence'] = sentence_str

        if len(lemma_sense):
            text_lemmas += lemma_sense
    return pd.DataFrame(text_lemmas)


if __name__ == "__main__":
    datapath = os.path.join("..", "data", "word_sense_disambigation_corpora")
    #1. Get everything into CSV format by lemma
    #2. Change Senses to WN senses
    corpus_path = os.path.join(datapath, "masc")
    wn_path_alg = os.path.join(datapath, "algorithmic_map.txt")
    wn_path_man = os.path.join(datapath, 'manual_map.txt')
    sense_map = pd.concat([pd.read_csv(wn_path_alg, sep = '\t', header = None), pd.read_csv(wn_path_man, sep = '\t', header = None)])
    sense_map.columns = ['NOAD', 'WordNet']
    result = list(Path(corpus_path).rglob("*.xml"))
    parsed_xml = [parse_xml(r, sense_map) for r in result]
    pd.concat(parsed_xml).to_csv("../data/google_ws_data_wn_synsets.csv")
