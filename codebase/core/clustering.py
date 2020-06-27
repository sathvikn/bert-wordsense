import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

#Plots
def plot_embeddings(e, sense_indices, sense_names, word_name, savefile = False):
    """
    Plots the embeddings with t-SNE, colored by sense

    Inputs:
    e- list of word embeddings
    sense_indices- list of indices specifying where the embeddings for each sense are
    sense_names- list of WordNet Synsets for each unique sense
    word_name- string in format word.pos
    savefile- if True, saves file as PNG with path ../data/clustering_results/word_pos/tsne.png

    Output
    sense_dict = {
        Synset for sense : t-SNE coordinates in 2D list
    }
    """
    assert len(sense_indices) == len(sense_names)
    as_arr = np.asarray(convert_embeddings(e))
    dim_red = TSNE()
    tsne_results = dim_red.fit_transform(as_arr)
    num_senses = len(sense_indices)
    results_for_sense = []

    results_for_sense.append(tsne_results[:sense_indices[0] - 1])
    for i in np.arange(len(sense_indices) - 1):
        start = sense_indices[i]
        end = sense_indices[i + 1]
        results_for_sense.append(tsne_results[start:end])
    
    sense_dict = {}
    for i in range(num_senses):
        if i == 2:
            plt.scatter(results_for_sense[i][:,0], results_for_sense[i][:,1], label = sense_names[i], color = 'violet')
        else:
            plt.scatter(results_for_sense[i][:,0], results_for_sense[i][:,1], label = sense_names[i])
        sense_dict[sense_names[i]] = results_for_sense[i]

    plt.title("BERT Embeddings for Senses of the Word \"" + word_name + "\" ")
    plt.legend()
    if savefile: #TODO see if we can make this a fn?
        word_token, word_pos = get_name(word_name), get_pos(word_name)
        path = os.path.join('..', 'data', 'clustering_results', word_token + '_' + word_pos, 'tsne.png')
        plt.savefig(path)
        plt.clf()
        plt.cla()

    return sense_dict

def plot_dendrogram(embed_data, color_dict, label_dict, savefile = False):
    """
    Plots a dendrogram of the BERT embeddings using single linkage agglomerative clustering, where the nodes are colored by sense
    
    embed_data = {
        'lemma': word.pos
        'embeddings': list of 768 x 1 PyTorch tensors representing vectorized representations of the lemma in a sentence in SEMCOR
        'original_sentences': List of strings representing sentences where each embedding came from
        'sense_names': List of Synsets, one for each unique sense of the word where we have embeddings
        'sense_labels': List of strings representing WordNet senses (name.pos.number), that specifies the sense of the lemma used in each sentence 
    }
    color_dict = {
        sense_name: color_str #C1, C2, C3 in matplotlib
    }
    label_dict = {
        index: {'color': color tag, 'label': sense label}
    }
    savefile- if True, saves file as PNG with path ../data/clustering_results/word_pos/dendrogram.png
    """
    embeds = convert_embeddings(embed_data['embeddings'])
    Z = linkage(embeds, method = 'single', metric = 'cosine')
    #plt.figure(figsize = (20, 8)) # for Jupyter plotting
    plt.figure(figsize = (9, 6)) #to plot on PDF
    dendrogram(Z, labels = embed_data['sense_labels'], link_color_func=lambda k: 'gray')
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(color_dict[lbl.get_text()])

    leg_patches = [mpatches.Patch(color = label_dict[i]['color'],
                                label = label_dict[i]['label']) for i in np.arange(len(label_dict))]
    plt.legend(handles=leg_patches)
    plt.title("Nearest Neighbor Dendrogram for BERT Embeddings of " + embed_data['lemma'] + " in SEMCOR")
    if savefile:
        word_name = embed_data['lemma']
        word_token, word_pos = get_name(word_name), get_pos(word_name)
        path = os.path.join('..', 'data', 'clustering_results', word_token + '_' + word_pos, 'dendrogram.png')
        plt.savefig(path)
        plt.clf()
        plt.cla()

#Utility functions
def recode_labels(true_labels):
    """
    Input:
    true_labels- list of strings showing WordNet senses

    Output:
    senses_as_nums- senses encoded as numbers
    """
    seen = {}
    senses_as_nums = []
    label_num = 0
    for l in true_labels:
        if l not in seen:
            seen[l] = label_num
            label_num += 1
        senses_as_nums.append(seen[l])
    return senses_as_nums

def convert_embeddings(embeds):
    """
    converts lists/PyTorch tensors to Numpy arrays for embeddings
    """
    if type(embeds[0]) == list: #Python list
        embeds = [np.array(v) for v in embeds]
    else: #PyTorch tensors
        embeds = [v.numpy() for v in embeds]
    return embeds

#Had issues importing these from semcor_bert_pipeline
def get_name(lem):
    if type(lem) != str:
        return lem.name()
    else:
        return lem.split('.')[0]

def get_pos(lem):
    lem = str(lem)
    try:
        return lem.split('.')[1]
    except:
        return 'No POS'


def create_dendrogram_colors(senses):
    """
    Input:
    senses- list of unique sense names

    Output:
    color_dict = {
        sense_name: color_str #C1, C2, C3 in matplotlib
    }
    label_dict = {
        index: {'color': color tag, 'label': sense label}
    }
    """
    color_dict = {}
    label_dict = {}
    for i in range(len(senses)):
        mplotlib_color = 'C' + str(i)
        sense = senses[i] #get_name issue?
        color_dict[sense] = mplotlib_color
        label_dict[i] = {'color': mplotlib_color, 'label': sense}
    return color_dict, label_dict


"""
==================
Deprecated, tried to run PCA on embeddings/compute the Rand index from Gaussian mixture models/compare this to t-SNE
==================
"""
def tsne_rand(pipeline_output):
    #Only works for 2-3 components
    results_for_word = []
    numpy_embeds = convert_embeddings(pipeline_output['embeddings'])
    for c in range(2, 4):
        tsne = TSNE(n_components = c)
        tsne_results = tsne.fit_transform(numpy_embeds)
        true_labels = recode_labels(pipeline_output['sense_labels'])
        num_senses = len(set(true_labels))
        gmm_results = gmm_rand(tsne_results, num_senses, true_labels)
        results_for_word.append({'Lemma': pipeline_output['lemma'], 'Principle Components': c,
                                 'WordNet Mean': gmm_results['GMM'][0], 'WordNet SD': gmm_results['GMM'][1], 
        'Random Mean': gmm_results['Random'][0],
        'Random SD': gmm_results['Random'][1]})
    return results_for_word

def process_pca(pca_results, num_comps, word, pos):
    results_for_word = []
    for c in num_comps:
        results_for_word.append({'Lemma': word + '.' + pos, 'Principle Components': c, 'WordNet Mean': np.mean(pca_results[c]['gmm_raw_aris']),
         'WordNet SD': np.std(pca_results[c]['gmm_raw_aris']), 'Random Mean': np.mean(pca_results[c]['random_baseline_raw_ari']),
          'Random SD': np.std(pca_results[c]['random_baseline_raw_ari'])})
    return results_for_word

def plot_pca_ev(comp_range, embeddings, lemma):
    embeddings = np.transpose(convert_embeddings(embeddings))
    ev_ratios = [sum(PCA(n_components = c).fit(embeddings).explained_variance_ratio_) for c in comp_range]
    plt.plot(comp_range, ev_ratios)
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.title("PCA on BERT embeddings of " + lemma)

def pca(embeddings, num_comps):
    embeds = convert_embeddings(embeddings)
    return sum(PCA(n_components = num_comps).fit(embeds).explained_variance_ratio_)

def plot_pca_grid(df, compare_column):
    fig = plt.figure(figsize = (10, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 10):
        ax = fig.add_subplot(3, 3, i)
        pcs = i + 1
        with_pcs = df[df['n_pcs'] == pcs]
        ax.scatter(with_pcs[compare_column], with_pcs['pct_var_explained'], alpha = 0.5)
        ax.title.set_text(str(pcs) + " PCs")
    plt.suptitle(compare_column + " vs. % Variance Explained")

def plot_gmm_rand_indices(embedding_data, comp_range, save_img = False, save_json = False):
    #Plots Rand Index means and SDs over 1000 GMM fits
    # returns dict of format: {PCA components: {GMM ARI, Random Baseline ARI}}
    embeddings = embedding_data['embeddings']
    lemma = embedding_data['lemma']
    senses = embedding_data['sense_labels']
    num_senses = len(set(senses))
    true_labels = recode_labels(senses)
    gmm_wn_means = []
    gmm_wn_sds = []
    gmm_random_means = []
    gmm_random_sds = []
    raw_results = {}

    for c in comp_range:
        pca_result = pca(embeddings, c)
        results = gmm_rand(pca_result, num_senses, true_labels)
        gmm_wn_means.append(results['GMM'][0])
        gmm_wn_sds.append(results['GMM'][1])
        gmm_random_means.append(results['Random'][0])
        gmm_random_sds.append(results['Random'][1])
        raw_results[c] = {'gmm_raw_aris': results['gmm_raw'], 'random_baseline_raw_ari': results['random_raw']}
    """
    plt.errorbar(comp_range, gmm_wn_means, yerr = gmm_wn_sds, label = "WordNet Senses")
    plt.errorbar(comp_range, gmm_random_means, yerr = gmm_random_sds, label = "Random Baseline")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Rand Index")
    plt.title("Rand Indices for PCA decomposition of " + lemma)
    plt.legend(title = 'Ground Truth')
    """
    word_token, word_pos = get_name(lemma), get_pos(lemma)
    if save_img:
        img_path = os.path.join('..', 'data', 'clustering_results', word_token + '_' + word_pos, 'gmm_evr.png')
        plt.savefig(img_path)
        plt.clf()
        plt.cla()
    if save_json:
        json_path = os.path.join('..', 'data', 'clustering_results', word_token + '_' + word_pos, 'gmm_results.json')
        with open(str(json_path), 'w') as path:
            json.dump(raw_results, path)

    return raw_results

def gmm_rand(clustered_result, wn_labels, true_labels):
    ari_gmm = []
    ari_random = []
    for _ in range(1000):
        gmm = GaussianMixture(n_components = wn_labels)
        gmm_preds = gmm.fit_predict(clustered_result)
        ari_gmm.append(adjusted_rand_score(gmm_preds, true_labels))
        random_clusters = np.random.choice(np.unique(true_labels), len(true_labels))
        ari_random.append(adjusted_rand_score(gmm_preds, random_clusters))
    return {'GMM': [np.mean(ari_gmm), np.std(ari_gmm)], 'Random': [np.mean(ari_random), np.std(ari_random)],
    "gmm_raw": ari_gmm, 'random_raw': ari_random}
