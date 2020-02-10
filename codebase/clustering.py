import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score


def plot_embeddings(e, sense_indices, sense_names, word_name):
    assert len(sense_indices) == len(sense_names)
    as_arr = np.asarray([t.numpy() for t in e])
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
    return sense_dict

#clustering/viz
def plot_dendrogram(embed_data, color_dict, label_dict):
    #color_dict is of format: {sense_name: color_str...}
    #label_dict is of format: {index: {'color': char, 'label': sense label}}
    
    embeds = [v.numpy() for v in embed_data['embeddings']]
    Z = linkage(embeds, method = 'single', metric = 'cosine')
    plt.figure(figsize = (20, 8))
    dendrogram(Z, labels = embed_data['sense_labels'], link_color_func=lambda k: 'gray')
    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(color_dict[lbl.get_text()])

    leg_patches = [mpatches.Patch(color = label_dict[i]['color'],
                                label = label_dict[i]['label']) for i in np.arange(len(label_dict))]
    plt.legend(handles=leg_patches)
    plt.title("Nearest Neighbor Dendrogram for BERT Embeddings of " + embed_data['lemma'] + " in SEMCOR")

def plot_pca_ev(comp_range, embeddings, lemma):
    embeddings = np.transpose(np.array([v.numpy() for v in embeddings]))
    ev_ratios = [sum(PCA(n_components = c).fit(embeddings).explained_variance_ratio_) for c in comp_range]
    plt.plot(comp_range, ev_ratios)
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.title("PCA on BERT embeddings of " + lemma)

def pca(embeddings, num_comps):
    embeds = np.transpose(np.array([v.numpy() for v in embeddings]))
    return PCA(n_components = num_comps).fit(embeds).components_.T

def plot_gmm_rand_indices(embedding_data, comp_range):
    #Plots Rand Index 
    embeddings = embedding_data['embeddings']
    lemma = embedding_data['lemma']
    senses = embedding_data['sense_labels']
    num_senses = len(set(senses))
    true_labels = recode_labels(senses)
    gmm_wn_means = []
    gmm_wn_sds = []
    gmm_random_means = []
    gmm_random_sds = []
    for c in comp_range:
        pca_result = pca(embeddings, c)
        results = gmm_rand(pca_result, num_senses, true_labels)
        gmm_wn_means.append(results['GMM'][0])
        gmm_wn_sds.append(results['GMM'][1])
        gmm_random_means.append(results['Random'][0])
        gmm_random_sds.append(results['Random'][1])
    plt.errorbar(comp_range, gmm_wn_means, yerr = gmm_wn_sds, label = "WordNet Senses")
    plt.errorbar(comp_range, gmm_random_means, yerr = gmm_random_sds, label = "Random Baseline")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Rand Index")
    plt.title("Rand Indices for PCA decomposition of " + lemma)
    plt.legend(title = 'Ground Truth')
    return gmm_wn_means

def recode_labels(true_labels):
    seen = {}
    senses_as_nums = []
    label_num = 0
    for l in true_labels:
        if l not in seen:
            seen[l] = label_num
            label_num += 1
        senses_as_nums.append(seen[l])
    return senses_as_nums

def gmm_rand(pca_result, wn_labels, true_labels):
    ari_gmm = []
    ari_random = []
    for _ in range(1000):
        gmm = GaussianMixture(n_components = wn_labels)
        gmm_preds = gmm.fit_predict(pca_result)
        ari_gmm.append(adjusted_rand_score(gmm_preds, true_labels))
        random_clusters = np.random.choice(np.unique(true_labels), len(true_labels))
        ari_random.append(adjusted_rand_score(gmm_preds, random_clusters))
    return {'GMM': [np.mean(ari_gmm), np.std(ari_gmm)], 'Random': [np.mean(ari_random), np.std(ari_random)]}        
