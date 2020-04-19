import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def parse_xml(fpath):
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
                lemma_sense.append({'lemma': t['lemma'], 'pos': t['pos'], 'sense': t['sense']})

        for l in lemma_sense:
            l['sentence'] = sentence_str
        text_lemmas += lemma_sense
    return pd.DataFrame(text_lemmas)


if __name__ == "__main__":
    datapath = os.path.join("..", "data", "word_sense_disambigation_corpora")
    #1. Get everything into CSV format by lemma
    #2. Change Senses to WN senses
    corpus_path = os.path.join(datapath, "masc")
    result = list(Path(corpus_path).rglob("*.xml"))
    
    parsed_xml = [parse_xml(r) for r in result]
