import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage 


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
def plot_dendrogram(embeds, color_dict, label_dict, word_pos):
    embeds = [v.numpy() for v in embeds]
    Z = linkage(embeds, method = 'single', metric = 'cosine')
    plt.figure(figsize = (20, 8))
    dendrogram(Z, labels = label_dict, link_color_func=lambda k: 'gray')

    ax = plt.gca()
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        lbl.set_color(color_dict[lbl.get_text()])
    
    leg_patches = [mpatches.Patch(color = label_dict[i]['color'],
                                  label = label_dict[i]['label']) for i in np.arange(len(label_dict))]
    plt.legend(handles=leg_patches)
    plt.title("Nearest Neighbor Dendrogram for BERT Embeddings of " + word_pos + " in SEMCOR")
