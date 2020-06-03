import numpy as np
import torch
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from . import semcor_bert_pipeline
from . import analysis
from scipy import stats
from adjustText import adjust_text
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#Distance metrics
def euc_dist(v1, v2):
    """
    Computes Euclidean distance between two arrays (converts from PyTorch first)
    """
    if type(v1) == torch.Tensor:
        v1 = v1.numpy()
    if type(v2) == torch.Tensor:
        v2 = v2.numpy()
    return np.sqrt(np.sum((v1 - v2)**2))

def find_closest_distance(e1_lst, e2_lst, fn):
    #fn can be either np.mean or minimum
    return fn([min([euc_dist(e1, e2) for e2 in e2_lst]) for e1 in e1_lst])

def centroid(arr):
    """
    Find the centroid of a Numpy array of embeddings (n x 768), where n is the total number of words
    """
    length, dim = arr.shape
    return np.array([np.sum(arr[:, i])/length for i in range(dim)])

def lst_to_np(arr):
    """
    Input: arr- list of PyTorch tensor objects

    Output: Numpy array of the tensors, themselves converted to arrays
    """
    return np.array([t.numpy() for t in arr])

def cosine_sim(v1, v2):
    """
    Takes the cosine similarity between two Numpy arrays of embeddings (v1 and v2)
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cs_centroids(s1, s2):
    """
    Takes the cosine similarity of the centroids of two Numpy matrices of embeddings (s1 and s2)
    """
    return cosine_sim(centroid(s1), centroid(s2))

def dist_centroids(s1, s2):
    """
    Euclidean distance between the centroids of two Numpy matrices of embeddings
    """
    return euc_dist(centroid(s1), centroid(s2))

def cosine_sim_mtx(word, pos, sel_senses = [], use_masc = False, normalize = False):
    """
    Inputs:
    word- string, lemma name
    pos- string, part of speech (n, v, s)
    sel_senses- list of senses to compute similarity between (defaults to all)
    use_masc- if Google MASC corpus should be included
    normalize- if True, return the normalized cosine distance matrix 

    Output:

    result_mtx- matrix of pairwise cosine similarities for the senses of type word.pos
    sel_senses- list of senses compared in the matrix (defaults to all senses)
    """

    #Loads from filesystem
    data = semcor_bert_pipeline.load_data(word, pos, 'semcor')
    word_embeddings = data['embeddings']
    sense_labels = data['sense_labels']

    if use_masc:
        try:
            masc_data = semcor_bert_pipeline.load_data(word, pos, 'masc')
            word_embeddings += masc_data['embeddings']
            sense_labels += masc_data['sense_labels']
        except:
            pass

#Gathering embeddings
    embeddings_by_sense = {}
    word_embeddings = np.array([np.array(e) for e in word_embeddings])
    if not len(sel_senses):
        strip_synset = lambda s: s.strip("Synset()").strip("'")
        sel_senses = [strip_synset(i) for i in data['sense_names']]
    for s in sel_senses:
        embeddings_by_sense[s] = word_embeddings[np.argwhere(np.array(sense_labels) == s).flatten()]
    result_mtx = []

    #For each pair of senses, compute cosine similarity between centroids
    for i in sel_senses:
        row = []
        for j in sel_senses:
            dist = cs_centroids(embeddings_by_sense[i], embeddings_by_sense[j])
            row.append(dist)
        result_mtx.append(np.asarray(row))
    result_mtx = np.asarray(result_mtx)
    if normalize:
        return normalize_cos_dist(result_mtx), sel_senses
    return result_mtx, sel_senses

def normalize_cos_dist(cs_mtx):
    """
    Input: cs_mtx is a nxn dimensional matrix of pairwise cosine similarities for a type
    Output: Distance matrix where entries are divided by largest distance 
    """
    max_value = np.max(1 - cs_mtx)
    return (1 - cs_mtx) / max_value

#Multiclass Logistic Regression

def k_fold_cv(x, y, k = 5):
    """
    Input:
    x- data (n x 768 numpy matrix of embeddings)
    y- labels (length-n list of labels, coded as integers)
    k- number of folds

    Output:
    f- list of F1 scores (1 per fold)
    acc- list of accuracies (1 per fold)
    incorrect_indices- indices of incorrect classifications
    confusion_matrices- list of confusion matrices (1 per fold)
    """
    kf = KFold(n_splits = k, shuffle = True)
    f = []
    acc = []
    incorrect_indices = []
    confusion_matrices = []
    for train_index, test_index in kf.split(x):
        model = LogisticRegression(penalty = 'l1', multi_class = 'multinomial', solver = 'saga', max_iter = 5000)
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        #print(classification_report(y_test, test_pred))
        f.append(f1_score(y_test, test_pred, average = "weighted"))
        acc.append(accuracy_score(y_test, test_pred))
        confusion_matrices.append(confusion_matrix(y_test, test_pred, labels = np.unique(y)))
        incorrect_indices +=[i[0] for i in np.argwhere(y_test != test_pred) if i[0] not in incorrect_indices]
    return f, acc, incorrect_indices, confusion_matrices


def binary_logistic(word_pos, target_senses):
    """
    Input:
    word_pos- string with format word.pos
    target_senses- list of two senses

    Output:
    {
    'model': Scikit Learn LogisticRegression object fitted to classifying the pair of embeddings , 
    'data': n x 768 numpy matrix of embeddings,
    'labels': length n array of senses each vector corresponds to, 
    'transformed_labels': Numpy array of labels encoded as integers, 
    'sentences': Original sentences using the pair of senses
    }
    """
    word, pos = word_pos.split('.')
    data = semcor_bert_pipeline.load_data(word, pos, 'semcor')
    #masc_data = semcor_bert_pipeline.load_data(word, pos, 'masc')
    le = LabelEncoder()
    sense_labels = data['sense_labels']#  + masc_data['sense_labels']
    le.fit(target_senses)
    sense_indices = [i for i in range(len(sense_labels)) if sense_labels[i] in target_senses]
    x = np.array(data['embeddings'])[sense_indices] #+ masc_data['embeddings'])[sense_indices]
    labels = np.array(data['sense_labels'])[sense_indices] #+ masc_data['sense_labels'])[sense_indices]
    y = le.transform(labels)
    model = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter = 5000)
    model.fit(x, y)
    return {'model': model, 'data': x, 'labels': np.asarray(labels), 'transformed_labels': y, 'sentences': np.asarray(data['original_sentences'])}

def misclassified_sentences(model_data, incorrect_indices):
    """
    Inputs:
    model_data- output from binary_logistic or logistic_cv
    incorrect_indices- indices of elements that were incorrectly classified

    Output:
    2 column Pandas Dataframe containing the true label and the sentence it came from
    """
    sense_names, sentences = model_data['labels'][incorrect_indices], model_data['sentences'][incorrect_indices]
    return pd.DataFrame({'true_label': sense_names, 'sentences': sentences})

def logistic_cv(lemma, sel_senses = [], use_masc = True, delim = '.'):
    """
    Inputs:
    lemma- String of word[delim]pos
    sel_senses- senses that will be considered (all senses by default)
    use_masc- whether the Google MASC dataset will be loaded
    delim- character separating word and PoS

    Output:
    {'model': Scikit Learn LogisticRegression object fitted to classifying the senses of the full set of embeddings with L1 regularization,
     "data": n x 768 matrix of embeddings for a word, 
     "labels": sense for each embedding,
     "acc": accuracies across 5 fold cross validation,
     "f1": F1 scores across 5 fold cross validation, 
    'incorrect_indices': list of indices of items in data that were incorrectly classified during all cross validation folds, 
    'sentences': list of original sentences embeddings were derived from,
    'confusion_matrices': list of confusion matrices from each fold of cross validation, 
    'weights': (number of senses - 1) x 768 array, showing weights of pairwise sense classification from model
    }
    """
    name, pos = lemma.split(delim)
    data = semcor_bert_pipeline.load_data(name, pos, 'semcor')
    embeddings = data['embeddings']
    sense_labels = np.asarray(data['sense_labels'])
    strip_synset = lambda s: s.strip("Synset()").strip("'")
    target_senses = [strip_synset(i) for i in data['sense_names']]
    sentences = np.asarray(data['original_sentences'])
    try:
        masc_data = semcor_bert_pipeline.load_data(name, pos, 'masc')
        embeddings += masc_data['embeddings']
        sense_labels += masc_data['sense_labels']
    except:
        pass
    le = LabelEncoder()
    le.fit(target_senses)
    x = np.asarray(embeddings)
    if len(sel_senses):
        sense_indices = [i for i in range(len(sense_labels)) if sense_labels[i] in sel_senses]
        x = x[sense_indices]
        sense_labels = sense_labels[sense_indices]
        sentences = sentences[sense_indices]
    y = le.transform(sense_labels)

    model = LogisticRegression(penalty = 'l1', solver = 'saga', max_iter = 20000)
    model.fit(x, y)
    #weight_values, weight_indices = nonzero_weights(model)
    f_scores, accuracies, wrong_indices, confusion_matrices = k_fold_cv(x, y, k = 5, labels = target_senses)
    return {'model': model, "data": x, "labels": np.asarray(sense_labels), "acc": accuracies, "f1": f_scores, 
            'incorrect_indices': wrong_indices, 'sentences': sentences,
           'confusion_matrices': confusion_matrices, 'weights': model.coef_}

def plot_confusion_mtx(word_matrices, senses, with_dendrogram = False):

    """
    Inputs:
    word_matrices- Array of confusion matrices for a type
    senses- list of senses that were classified
    with_dendrogram- whether the matrix should be organized according to order (sns.clustermap)
    
    Plots aggregated confusion matrix for type (P(class | true label)), columns are true labels, rows are predicted
    """
    agg_confusion = np.sum(np.asarray(word_matrices), axis = 0)
    agg_confusion = np.nan_to_num(agg_confusion / np.sum(agg_confusion, axis = 0))
    if with_dendrogram:
        sns.clustermap(pd.DataFrame(agg_confusion, columns = senses , index = senses), method="single", 
                   figsize = (6, 6), cmap = 'mako', annot = True, vmin=0, vmax=1, cbar_pos = None)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        return
    fig, ax = plt.subplots()
    im = plt.imshow(agg_confusion)
    analysis.annotate_mtx(agg_confusion, im, ax, senses)


#Deprecated, measuring Rand index of Gaussian Mixture Models
def get_word_senses_ri(folder_name):
    embed_fpath = os.path.join('..', 'data', 'pipeline_results', folder_name + '.json')
    with open(embed_fpath, 'r') as embeds:
        embed_json = json.load(embeds)
    num_senses = len(embed_json['sense_names'])
    gmm_fpath = os.path.join('..', 'data', 'clustering_results', folder_name, 'gmm_results.json')
    with open(gmm_fpath, 'r') as ri:
        ri_pca = json.load(ri)
    return num_senses, ri_pca

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

def check_for_embedding_data(word, pos):
    fname = word + '_' + pos + '.json'
    if fname in os.listdir(os.path.join('..', 'data', 'pipeline_results')):
        return 1
    else:
        return 0

def plot_metric_ri(x_col, df, method, dims):
    plt.subplots()
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
                if d == '2D':
                    fn(v, two_pc, m, d)
                else:
                    fn(v, three_pc, m, d)

def get_sample(two_pc, three_pc, num_words):
    two_sample = two_pc.sample(num_words)
    three_sample = three_pc[three_pc['Lemma'].isin(two_sample['Lemma'])]
    return two_sample, three_sample

def scatter_gmm_results_text(x_col, df, method, dims):
    #plt.scatter(ent, df['Random Mean_' + method], label = 'random')
    x = df[x_col]
    y = df['WordNet Mean_' + method]
    labels = df['Lemma']
    #plt.subplots()
    plt.figure(figsize = (10, 8))
    
    plt.scatter(x, y)
    texts = []
    i = 0
    for x, y, s in zip(x, y, labels):
        texts.append(plt.text(x, y, s))
    plt.xlabel(x_col)
    plt.ylabel("Rand Index")
    plt.title("Rand Scores for GMMs fitted to " + method.upper() + " Embeddings " + "vs. " + x_col + '(' + dims + ')')
    
    adjust_text(texts, force_points=0.2, force_text=0.2,
           expand_points=(1, 1), expand_text=(1, 1),
            arrowprops=dict(arrowstyle="-", color='black', lw=0.5))


def get_corr_words():
    df = pd.read_csv('..data/semcor_sparsity.csv')
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

def nonzero_weights(model):
    weights = model.coef_[0]
    nonzero_indices = np.where(weights != 0)[0]
    return weights[nonzero_indices], nonzero_indices
