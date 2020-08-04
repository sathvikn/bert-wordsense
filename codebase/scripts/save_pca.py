import os
import sys
import numpy as np
from sklearn.decomposition import PCA
sys.path.append("..")
from core.semcor_bert_pipeline import write_json, load_data


data_path = os.path.join('..', 'data', 'pipeline_results', 'semcor')
os.system('mkdir ../data/pipeline_results/semcor_pca')
for f in os.listdir(data_path):
    word = '_'.join(f.split("_")[:-1])
    pos = f.split("_")[-1].split(".")[0]
    try:
        type_data = load_data(word, pos, 'semcor')
        type_pca = {}
        if type_data['embeddings']:
            pca = PCA(n_components=22)
            data = type_data['embeddings']
            pca_embeds = pca.fit_transform(data)
            type_pca['embeddings'] = pca_embeds.tolist()
            type_pca['original_sentences'] = type_data['original_sentences']
            type_pca['sense_labels'] = type_data['sense_labels']
        write_json(type_pca, word, pos, 'semcor_pca')
    except:
        pass
