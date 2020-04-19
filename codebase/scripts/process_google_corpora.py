import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


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
                wn_sense = sense_map[sense_map['NOAD'] == t['sense']]['WordNet']
                lemma_sense.append({'lemma': t['lemma'], 'pos': t['pos'], 'sense': wn_sense})

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
    wn_mapping_path = os.path.join(datapath, "algorithmic_map.txt")
    sense_map = pd.read_csv(wn_mapping_path, sep = '\t', header = None)
    sense_map.columns = ['NOAD', 'WordNet']
    result = list(Path(corpus_path).rglob("*.xml"))
    parsed_xml = [parse_xml(r, sense_map) for r in result]
    pd.concat(parsed_xml).to_csv("../data/google_ws_data.csv")
