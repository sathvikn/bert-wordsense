import pandas as pd
from semcor_bert_pipeline import *
from clustering import *


def run_clustering(word, pos, model):
    word_results = run_pipeline(word, pos, model)
    tsne_results = plot_embeddings(word_results['embeddings'], word_results['sense_indices'], word_results['sense_names'], word_results['lemma'], savefile = True)
    color_dict, label_dict = create_dendrogram_colors(word_results['sense_names'])
    plot_dendrogram(word_results, color_dict, label_dict, savefile = True)
    raw_json = plot_gmm_rand_indices(word_results, range(2, 40))

if __name__ == '__main__':
    model = initialize_model()
    sparse_senses = pd.read_csv('data/semcor_sparsity.csv')
    for i in len(sparse_senses.index):
        row = sparse_senses.iloc[i]
        word, pos = row['word'], row['pos']
        run_clustering(word, pos, model)
