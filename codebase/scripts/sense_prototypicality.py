import os
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from core.semcor_bert_pipeline import load_data
from core.metrics import cosine_sim, centroid

"""
Spec: for each JSON in the semcor folder, create a CSV with following schema:
type.csv -> sense | sentence | cosine similarity to centroid

Approach
For each type:
compute centroids of all the senses (store them in a dict)
For each sentence, find the type's cosine similarity to its sense's centroid
How -> index into embedding list, get sense for embedding at that index, query the dict that way
Once you get the centroid, save the results as a row of the dataframe
"""

def centroid_dict(sense_data, n):
    type_centroids = {}
    for s in np.unique(sense_data['sense_labels']):
        sense_indices = np.argwhere(np.array(sense_data['sense_labels']) == s).flatten()
        embeds_for_sense = np.array(sense_data['embeddings'])[sense_indices]
        type_centroids[s] = centroid(embeds_for_sense)
    return type_centroids

path = "../data/pipeline_results/select_weights"
for f in os.listdir(path):
    word = '_'.join(f.split("_")[:-1])
    pos = f.split("_")[-1].split(".")[0]
    try:
        type_data = load_data(word, pos, 'select_weights')
        orig_data = load_data(word, pos, 'semcor')
        if type_data['embeddings']:
            num_instances = len(type_data['embeddings'])
            centroids = centroid_dict(type_data, num_instances)
            type_sims = []
            for i in range(num_instances):
                embed, sense = type_data['embeddings'][i], type_data['sense_labels'][i]
                instance_data = {"sense": sense, "sentence": orig_data['original_sentences'][i]}
                for s in centroids:
                    instance_data[s] = cosine_sim(embed, centroids[s])
                type_sims.append(instance_data)
        similarity_df = pd.DataFrame(type_sims)
        similarity_df.to_csv(os.path.join("../data/prototypicality_stimuli/", word + "_" + pos + ".csv"), index = False)
    except:
        pass

