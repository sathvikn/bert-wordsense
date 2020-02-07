import nltk
import torch
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
#from nltk.corpus import wordnet
from scipy.cluster.hierarchy import dendrogram, linkage 
import matplotlib.pyplot as plt
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.manifold import TSNE

#Selecting words from SEMCOR
class SemCorSelector:
    def __init__(self):
        self.semcor_sentences = nltk.corpus.semcor.sents()
        self.semcor_tagged_sents = nltk.corpus.semcor.tagged_sents(tag = 'sem')
        #assert len(self.semcor_sentences) == len(self.semcor_tagged_sents)
        self.num_sentences = len(self.semcor_sentences)
        self.tagged_word = []
        self.original_sent_for_word = []
        self.senses = []
        self.curr_word = ''
    
    def get_word_data(self, word, pos):
        self.curr_word = word + '.' + pos
        #self.semcor_for_word(word, pos)
        for i in range(self.num_sentences):
            s = self.semcor_tagged_sents[i]
            for tok in s:
                if type(tok) == nltk.tree.Tree:
                    lem = tok.label()
                    if type(lem) == nltk.corpus.reader.wordnet.Lemma:
                        if get_pos(lem) == pos and get_name(lem) == word:
                            self.tagged_word.append(self.semcor_tagged_sents[i])
                            self.original_sent_for_word.append(self.semcor_sentences[i])
                            #sense = (pos, get_sense_num(lem))
                            self.senses.append(lem.synset())

        #self.senses = [self.get_sense_for_sent(s, word) for s in self.tagged_word]
    
    def get_senses_for_curr_word(self):
        if len(self.senses):
            print("Senses for word", self.curr_word)
            return set(self.senses)
        else:
            print("No word has been searched for.")
    
    def get_selected_sense_sents(self, sel_senses):
        #sel_senses is a list of WordNet Synsets
        original_full = self.change_original(self.original_sent_for_word)
        last_ind_for_sense = []
        selected_origs = []
        selected_tagged = []

        for s in sel_senses:
            sense_indices = self.get_ind_for_sense(s)
            print("Number of sentences for sense", s, len(sense_indices))
            if len(last_ind_for_sense):
                last_ind_for_sense.append(last_ind_for_sense[-1] + len(sense_indices))
            else:
                last_ind_for_sense.append(len(sense_indices))
            selected_origs += [original_full[i] for i in sense_indices]
            selected_tagged += [self.tagged_word[i] for i in sense_indices]
        return selected_origs, selected_tagged, last_ind_for_sense     

    #def semcor_for_word(self, word, pos):
        #Orig, tagged, senses
        """
        word_indices = [i for i in np.arange(len(self.semcor_sentences)) if word in self.semcor_sentences[i]]
        tagged_word = [self.semcor_tagged_sents[i] for i in word_indices]
        original_sentences = [self.semcor_sentences[i] for i in word_indices]
        self.tagged_word = tagged_word
        self.original_sent_for_word = original_sentences

    def get_sense_for_sent(self, sent, word):
        for w in sent:
            try:
                if type(w) == nltk.tree.Tree:
                    this_word = w.leaves()[0]
                    if word == this_word:
                        return self.get_sense_pos(w)
            except:
                continue
    """

    def get_sense_pos(self, tree):
        lem = tree.label()
        #word = tree.leaves()[0]
        sense = str(lem).split('.')[2]
        pos = str(lem).split('.')[1]
        return (pos, sense)

    def change_original(self, original_s_list):
        return [' '.join(s) for s in original_s_list]

    def get_ind_for_sense(self, synset):
        return [i for i in np.arange(len(self.senses)) if self.senses[i] == synset]

def get_pos(lem):
    lem = str(lem)
    try:
        return lem.split('.')[1]
    except:
        return 'No POS'

def get_sense_num(lem):
    lem = str(lem)
    try:
        return lem.split('.')[2]
    except:
        return "No marked sense"

def get_name(lem):
    if type(lem) != str:
        return lem.name()
    else:
        return lem.split('.')[0]
    
#BERT
def preprocess(text, target_word): #should take in a SemCorSelector object too?
    #START and STOP tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    marked_text = "[CLS] " + text + " [SEP]" 
    tokenized_text = tokenizer.tokenize(marked_text)
    #Indices according to BERT's vocabulary
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #BERT can work with either 1 or 2 sentences, but for our purposes we're using one
    segments_ids = [1] * len(tokenized_text)
    l = nltk.stem.WordNetLemmatizer()
    try:
        target_token_index = tokenized_text.index(target_word)
    except:
        for i in np.arange(len(tokenized_text)):
            if l.lemmatize(tokenized_text[i]) == target_word:
                target_token_index = i
                break
    return (indexed_tokens, segments_ids, target_token_index)

def initialize_model():
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return model

def predict(indexed_tokens, segments_ids, model):
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    token_embeddings = torch.stack(encoded_layers, dim=0)
    return torch.squeeze(token_embeddings, dim=1)

def get_embeddings(data, model):
    indexed_tokens = data[0]
    segments_ids = data[1]
    target_token_index = data[2]
    all_embeddings = predict(indexed_tokens, segments_ids, model)
    return all_embeddings[:,target_token_index,:]

def get_raw_embeddings(word, sentences, model):
    preprocessed_texts = [preprocess(t, word) for t in sentences]
    raw_embeddings = [get_embeddings(t, model) for t in preprocessed_texts]
    return raw_embeddings

def take_layer(token_embed, layer_num):
    return token_embed[layer_num]

def sum_layers(token_embed, num_layers):
    #Takes in a [12 x 768] tensor, sums the vectors from the last num_layers to get the complete embedding
    sum_vec = torch.sum(token_embed[-num_layers:], dim=0)
    return sum_vec

def process_raw_embeddings(raw_embeds, layer, fn):
    #fn can either be take_layer or sum_layers
    return [fn(t, layer) for t in raw_embeds]

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

def find_closest_distance(e1_lst, e2_lst):
    return min([min([euc_dist(e1, e2) for e2 in e2_lst]) for e1 in e1_lst])

def euc_dist(v1, v2):
    if type(v1) == torch.Tensor:
        v1 = v1.numpy()
    if type(v2) == torch.Tensor:
        v2 = v2.numpy()
    return np.sqrt(np.sum((v1 - v2)**2))

def select_senses(senses, pos, reader, min_sents):
    sel_senses = []
    for s in senses:
        if len(reader.get_ind_for_sense(s)) > min_sents:
            sel_senses.append(s)
    return sel_senses
"""
def get_sense_num(senses):
    return [s[1] for s in senses]

"""
def get_tree_labels(sense_indices, sel_senses):
    tree_labels = []
    this_sense_indices = [0] + sense_indices
    for i in np.arange(len(sense_indices)):
        start = this_sense_indices[i]
        end = this_sense_indices[i + 1]
        tree_labels += (end - start) * [sel_senses[i]]
    return tree_labels

def run_pipeline(word, pos, model):
    print("Getting data from SEMCOR")
    semcor_reader = SemCorSelector()
    semcor_reader.get_word_data(word, pos)
    senses = semcor_reader.get_senses_for_curr_word()
    print("Getting sentences for relevant senses")
    sel_senses = select_senses(senses, pos, semcor_reader, 10)
    #print(sel_senses)
    sentences, trees, sense_indices = semcor_reader.get_selected_sense_sents(sel_senses)
    tree_labels = get_tree_labels(sense_indices, sel_senses)
    print("Generating BERT embeddings")
    raw_embeddings = get_raw_embeddings(word, sentences, model)
    summed_embeds = process_raw_embeddings(raw_embeddings, 4, sum_layers)
    """
        if plot:
            print("Plotting t-SNE")
            leg = get_sense_num(sel_senses)
            tsne_results = plot_embeddings(summed_embeds, sense_indices, leg, word)
            return summed_embeds, tree_labels, tsne_results
    """    
    return summed_embeds, tree_labels

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

