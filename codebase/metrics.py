import numpy as np
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from semcor_bert_pipeline import *
from scipy import stats

def euc_dist(v1, v2):
    if type(v1) == torch.Tensor:
        v1 = v1.numpy()
    if type(v2) == torch.Tensor:
        v2 = v2.numpy()
    return np.sqrt(np.sum((v1 - v2)**2))

def find_closest_distance(e1_lst, e2_lst, fn):
    #fn can be either np.mean or minimum
    return fn([min([euc_dist(e1, e2) for e2 in e2_lst]) for e1 in e1_lst])

def centroid(arr):
    arr = lst_to_np(arr)
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])

def lst_to_np(arr):
    return np.array([t.numpy() for t in arr])

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cs_centroids(s1, s2):
    return cosine_sim(centroid(s1), centroid(s2))

def dist_centroids(s1, s2):
    return euc_dist(centroid(s1), centroid(s2))

def get_word_senses_ri(folder_name):
    embed_fpath = os.path.join('data', 'pipeline_results', folder_name + '.json')
    with open(embed_fpath, 'r') as embeds:
        embed_json = json.load(embeds)
    num_senses = len(embed_json['sense_names'])
    gmm_fpath = os.path.join('data', 'clustering_results', folder_name, 'gmm_results.json')
    with open(gmm_fpath, 'r') as ri:
        ri_pca = json.load(ri)
    return num_senses, ri_pca

def check_for_embedding_data(word, pos):
    fname = word + '_' + pos + '.json'
    if fname in os.listdir(os.path.join('data', 'pipeline_results')):
        return 1
    else:
        return 0

def plot_metric_ri(x_col, df, method, dims):
    plt.scatter(df[x_col], df['Random Mean_' + method], label = 'random')
    plt.scatter(df[x_col], df['WordNet Mean_' + method], label = 'WN Senses', alpha = 0.3)
    plt.xlabel(x_col)
    plt.ylabel("Rand Index")
    plt.title("Rand Scores for GMMs fitted to " + method.upper() + " Embeddings " + "vs. " + x_col + '(' + dims + ')')
    plt.legend()

def plot_dim_metric_method_combos(two_pc, three_pc, fn):
    for d in ['2D', '3D']:
        for m in ['pca', 'tsne']:
            for v in ['entropy', 'num_senses']:
                plt.subplots()
                if d == '2D':
                    fn(v, two_pc, m, d)
                else:
                    fn(v, three_pc, m, d)

def scatter_gmm_results_text(x_col, df, x_metric, method, dims):
    #plt.scatter(ent, df['Random Mean_' + method], label = 'random')
    x = df[x_col]
    y = df['WordNet Mean_' + method]
    labels = df['Lemma']
    plt.figure(figsize = (10, 8))
    
    plt.scatter(x, y)
    texts = []
    i = 0
    for x, y, s in zip(x, y, labels):
        texts.append(plt.text(x, y, s))
    plt.xlabel(x_col)
    plt.ylabel("Rand Index")
    plt.title("Rand Scores for GMMs fitted to " + method.upper() + " Embeddings " + "vs. " + x_metric + '(' + dims + ')')
    
    adjust_text(texts, force_points=0.2, force_text=0.2,
           expand_points=(1, 1), expand_text=(1, 1),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5))


def get_corr_words():
    df = pd.read_csv('data/semcor_sparsity.csv')
    words_with_data = []
    for i in range(len(df.index)):
        row = df.iloc[i]
        word, pos = row['word'], row['pos']
        folder_name = word + '_' + pos
        #fname = word + '_' + pos + '.json'
        #dir_name = os.path.join("data", 'pipeline_results', word + '_' + pos + '.json')
        if check_for_embedding_data(word, pos):
            words_with_data.append(folder_name)
    return words_with_data

def compute_correlation(corr_dict, key):
    num_senses = corr_dict[key]['num_senses']
    gmm = corr_dict[key]['gmm_ari']
    random = corr_dict[key]['random_ari']
    corr_dict[key]['pearson_gmm'] = stats.pearsonr(num_senses, gmm)[0]
    corr_dict[key]['spearman_gmm'] = stats.spearmanr(num_senses, gmm)[0]
    corr_dict[key]['pearson_random'] = stats.pearsonr(num_senses, random)[0]
    corr_dict[key]['spearman_random'] = stats.spearmanr(num_senses, random)[0]

def plot_correlation(corr_dict, max_pcs):
    random_sp, random_pe, gmm_sp, gmm_pe = [], [], [], []
    for k in corr_dict:
        pc_data = corr_dict[k]
        random_sp.append(pc_data['spearman_random'])
        random_pe.append(pc_data['pearson_random'])
        gmm_sp.append(pc_data['spearman_gmm'])
        gmm_pe.append(pc_data['pearson_gmm'])
    num_pcs = np.arange(2, max_pcs)
    plt.figure(figsize = (8, 6))
    plt.subplot(1, 2, 1)
    plt.plot(num_pcs, random_sp, label = "Random Baseline")
    plt.plot(num_pcs, gmm_sp, label = "WordNet Senses")
    plt.xlabel("Principle Components of BERT Embeddings")
    plt.ylabel("Correlation Coefficient")
    plt.title("Spearman Correlation")
    plt.subplot(1, 2, 2)
    plt.plot(num_pcs, random_pe, label = "Random Baseline")
    plt.plot(num_pcs, gmm_pe, label = "WordNet Senses")
    plt.xlabel("Principle Components of BERT Embeddings")
    plt.ylabel("Correlation Coefficient")
    plt.title("Pearson Correlation")
    plt.legend()