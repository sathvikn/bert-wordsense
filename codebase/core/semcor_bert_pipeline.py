import json
import os
import nltk
import torch
import numpy as np
import copy
from transformers import BertTokenizer, BertModel, BertConfig

#Selecting words from SEMCOR
class SemCorSelector:

    """
    API to interface with NLTK's SEMCOR corpus reader.
    Query NLTK with get_word_data function, and store relevant information detailing the current type, sentences it is used in, and its senses
    """
    def __init__(self):
        """
        Loads SEMCOR data from NLTK, as well as state variables 
        """
        #Constants
        self.semcor_sentences = nltk.corpus.semcor.sents()
        self.semcor_tagged_sents = nltk.corpus.semcor.tagged_sents(tag = 'sem')
        #assert len(self.semcor_sentences) == len(self.semcor_tagged_sents)
        self.num_sentences = len(self.semcor_sentences) 
        self.tagged_word = [] #Sentences tagged with WordNet lemmas
        self.original_sent_for_word = [] #Untagged sentences
        self.senses = [] #List of WordNet senses (Synset objects)
        self.curr_word = '' #type as word.pos
    
    def get_word_data(self, word, pos):
        """
        Input: self- SemCorSelector object
        word- word name
        pos- part of speech (n, v, s)

        Output: update state with data for type:
        word identity
        tagged sentences
        untagged sentences
        senses
        """
        self.curr_word = word + '.' + pos
        #Reset state
        self.original_sent_for_word = []
        self.senses = []
        self.tagged_word = []
        #self.semcor_for_word(word, pos)
        for i in range(self.num_sentences):
            s = self.semcor_tagged_sents[i]
            for tok in s:
                if type(tok) == nltk.tree.Tree: #Gets all lemmas tagged with WordNet senses
                    lem = tok.label()
                    if type(lem) == nltk.corpus.reader.wordnet.Lemma: 
                        if get_pos(lem) == pos and get_name(lem) == word: 
                            #If it matches, add the lemma's sense and both untagged and tagged versions of the sentence to the current state
                            self.tagged_word.append(self.semcor_tagged_sents[i])
                            self.original_sent_for_word.append(self.semcor_sentences[i])
                            #sense = (pos, get_sense_num(lem))
                            self.senses.append(lem.synset())

        #self.senses = [self.get_sense_for_sent(s, word) for s in self.tagged_word]
    
    def get_senses_for_curr_word(self):
        """
        Input:
        self- SemCorSelector object
        
        Output:
        The type's senses present in SEMCOR (as a Set)
        """
        if len(self.senses):
            print("Senses for word", self.curr_word)
            return set(self.senses)
        else:
            print("No word has been searched for.")
    
    def get_selected_sense_sents(self, sel_senses):
        """
        Input:
        self- SemCorSelector object
        sel_senses- list of WordNet Synsets

        Output:
        selected_origs and selected_tags are sorted by sense in the order specified by sel_senses
        selected_origs- list of original sentences with the selected senses
        selected_tagged- list of tagged sentences with the selected senses
        last_ind_for_sense- list of the last index a sense is used in selected_origs and selected_tagged for each sense in sel_senses
        """
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

    def get_sense_pos(self, tree):
        """
        DEPRECATED, look at get_pos and get_sense_num
        Inputs:
        self- SemCorSelector object
        tree- NLTK tree object

        Output:
        tuple of the POS and sense number
        """
        lem = tree.label()
        #word = tree.leaves()[0]
        sense = str(lem).split('.')[2]
        pos = str(lem).split('.')[1]
        return (pos, sense)

    def change_original(self, original_s_list):
        """
        Inputs:
        self- SemCorSelector Object
        original_s_list- list of lists of tokens corresponding to sentences

        Output: 
        List of all sentences as strings
        """
        return [' '.join(s) for s in original_s_list]

    def get_ind_for_sense(self, synset):
        """
        Inputs:
        self- SemCorSelector object
        synset- WordNet Synset object

        Output:
        indices for sentences corresponding to selected sense
        """
        return [i for i in np.arange(len(self.senses)) if self.senses[i] == synset]
    
    def select_senses(self, min_sents):
        """
        Inputs:
        self- SemCorSelector object
        min_sents- integer

        Outputs:
        Senses for the current type which have more than min_sents instances in SEMCOR
        """
        sel_senses = []
        uq_senses = self.get_senses_for_curr_word()
        for s in uq_senses:
            if len(self.get_ind_for_sense(s)) > min_sents:
                sel_senses.append(s)
        return sel_senses

#Utility functions

def get_pos(lem, delim = '.'):
    """
    Inputs:
    lem- WordNet lemma
    delim- delimiting character (. or ,)

    Output:
    Part of speech for lem (n, v, s)
    """
    lem = str(lem)
    try:
        return lem.split(delim)[1]
    except:
        return 'No POS'

def get_sense_num(lem, delim = '.'):
    """
    DEPRECATED?
    Inputs: 
    lem- WordNet lemma
    delim- delimiting character (. or ,)

    Output:
    Number of sense in WordNet
    """
    lem = str(lem)
    try:
        return lem.split(delim)[2]
    except:
        return "No marked sense"

def get_name(lem, delim = '.'):
    """
    Inputs:
    lem- WordNet lemma
    delim- delimiting character (. or ,)

    Output:
    Name of sense in WordNet
    """
    if type(lem) != str:
        return lem.name()
    else:
        return lem.split(delim)[0]

def process_tree(tree, word, pos):
    """
    Inputs:
    tree- list of tokens representing sentence
    word- String, word form
    type- String, part of speech (n, v, s)

    Outputs:
    Tree processed as sentence for BERT, target word
    """
    text = []
    word_with_sense = ''
    for t in tree:
        if type(t) == list:
            text += t
        if type(t) == nltk.Tree:
            text += t.leaves() #Append the word itself to the text
            word_name = get_name(t.label())
            word_pos = get_pos(t.label())
            if word_pos == pos and word_name == word:
                word_with_sense = t.leaves()[0]
    return ' '.join(text), word_with_sense.lower()

#Interfacing with BERT
    
def preprocess(text, target_word):
    """
    Inputs:
    Results from process_tree

    Outputs:
    indexed_tokens- text tokenized for BERT
    segments_ids- list of 1s (just one sentence of context)
    target_token_index- index of the target word in the tokenized text
    """
    #START and STOP tokens
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    marked_text = "[CLS] " + text + " [SEP]" 
    tokenized_text = tokenizer.tokenize(marked_text)
    #Indices according to BERT's vocabulary
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #DEPRECATED w transformers lib
    #BERT can work with either 1 or 2 sentences, but for our purposes we're using one
    segments_ids = [1] * len(tokenized_text)
    target_token_index = tokenized_text.index(target_word.lower())
    return (indexed_tokens, segments_ids, target_token_index), (tokenized_text, target_word)

def initialize_model():
    """
    Loading the pretrained BERT model from Huggingface
    """
    model_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states = True,
                                          output_attentions = True)
    model = BertModel.from_pretrained('bert-base-uncased', config = model_config)
    model.eval()
    return model

def predict(indexed_tokens, segments_ids, model):
    """
    Input:
    indexed_tokens and segments_ids are results from preprocess
    model- pre-trained BERT model from Huggingface

    Output:
    Pytorch tensor with embeddings for sentence
        Dimensions- (768 x number_of_tokens x 12)
    """
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    #token_embeddings = token_embeddings.permute(1,0,2)
    attentions = format_attention(outputs[3])
    return token_embeddings, attentions

def format_attention(attention):
    #From https://github.com/jessevig/bertviz/blob/master/bertviz/util.py
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def get_model_output(data, model):
    """
    Input:
    data: Array of texts with format from preprocess
    model: Pre-trained BERT model

    Outputs: Embeddings for the target token, attention tensor
    """
    indexed_tokens = data[0]
    segments_ids = data[1]
    target_token_index = data[2]
    all_embeddings, attentions = predict(indexed_tokens, segments_ids, model)
    return all_embeddings[:,target_token_index,:], attentions

def get_activations_attns(word, pos, trees, model):
    """
    Input:
    word- string, word that is being passed in
    pos- string, part of speech (n, v, s)
    trees- List of tagged sentences in Semcor
    model- pre-trained BERT model from Huggingface

    Output:
    list of 12 x 768 vectors where each element represents the type word.pos used in each sentence in trees
    list of 12 x 12 x seq_length x seq_length attention matrices
    """
    text_and_word = [process_tree(t, word, pos) for t in trees]
    preprocessed_texts = [preprocess(t[0], t[1]) for t in text_and_word]
    indexed_tokens = [t[0] for t in preprocessed_texts]
    text_tokens = [t[1] for t in preprocessed_texts]
    model_outputs = [get_model_output(t, model) for t in indexed_tokens]
    raw_embeddings = [o[0] for o in model_outputs]
    attn_tensors = [o[1] for o in model_outputs]
    return raw_embeddings, attn_tensors, text_tokens

def take_layer(token_embed, layer_num):
    """
    Access one of the 12 columns of a word's activation matrix (token_embed) at index layer_num.
    Outputs a 768 dimensional vector representing the activation of the word at the layer specified by layer_num
    """
    return token_embed[layer_num]

def sum_layers(token_embed, num_layers):
    """
    Takes in a [12 x 768] tensor(token_embed) sums the vectors from the last num_layers to get the complete embedding
    """

    sum_vec = torch.sum(token_embed[-num_layers:], dim=0)
    return sum_vec

def process_raw_embeddings(raw_embeds, layer, fn):
    """
    Inputs: 
    raw_embeds- from get_raw_embeddings (list of 12 x 768 matrices)
    layer- index that we're selecting/summing over
    fn- can either be take_layer or sum_layers here, but any function that takes in a 12 x 768 matrix and an index works

    Outputs a list of 768-dimensional embeddings for a word
    """
    return [fn(t, layer) for t in raw_embeds]

def query_attention(attn, tokens, layer, target_token):
    #attn is a Pytorch tensor with dimensions Layers x Heads x Token x Words token attends to
    token_index = tokens.index(target_token)
    
    all_heads = attn[layer, :, token_index]
    return np.mean(all_heads.numpy(), axis = 0)

def process_raw_attentions(attns, tokens):
    """
    Input:
    attns- list of 12 x 12 x seq_len x seq_len tensors consisting of attention matrices for tokens
    tokens- list of tuples consisting of (tokenized text, target token)

    Output:
    attention_vectors- a list of dictionaries where the attention for the token is averaged across the heads for each layer
    """

    attention_vectors = []
    for i in range(len(attns)):
        sentence_attns = {}
        for layer in range(12):
            sentence_attns[layer] = query_attention(attns[i], tokens[i][0], layer, tokens[i][1])
        attention_vectors.append(sentence_attns)
    return attention_vectors

def get_sense_labels(sense_indices, sel_senses):
    """
    Input:
    sel_senses- ordered list of Synset objects representing senses of a type in SEMCOR
    sense_indices- list of indices saying where one sense starts and the other begins

    Output:
    tree_labels- list of strings formatted as name.number.pos, originally used for dendrograms/plotting

    """
    tree_labels = []
    this_sense_indices = [0] + sense_indices
    for i in np.arange(len(sense_indices)):
        start = this_sense_indices[i]
        end = this_sense_indices[i + 1]
        tree_labels += (end - start) * [sel_senses[i].name()]
    return tree_labels

#Full pipeline & interaction with filesystem

def write_json(results, word, pos, corpus_dir):

    """
    Saves results from run_pipeline as JSON with filepath ../data/pipeline_results/[corpus_dir]/[word]_[pos].json
    """
    if len(pos) != 1:
        pos = pos[0].lower()
    filename = word + '_' + pos + '.json'
    #Fix this later??
    with open(os.path.join('..', 'data', 'pipeline_results', corpus_dir, filename), 'w') as f:
        json.dump(results, f)

def run_pipeline(word, pos, model, min_senses = 10, savefile = False, dir_name = "semcor"):

    """
    Input:
    word- string, name of lemma
    pos- string, part of speech (n, v, s)
    model- pre-trained BERT model from Hugging Face
    min_senses- for a sense to be included, it must have at least min_senses instances in SEMCOR
    savefile- boolean, says whether we want to save the embeddings

    Output:
    result_dict = {
        'lemma': word.pos
        'embeddings': list of 768 x 1 PyTorch tensors representing vectorized representations of the lemma in a sentence in SEMCOR
        'original_sentences': List of strings representing sentences where each embedding came from
        'sense_names': List of Synsets, one for each unique sense of the word where we have embeddings
        'sense_labels': List of strings representing WordNet senses (name.pos.number), that specifies the sense of the lemma used in each sentence 
    }
    """
    print("Getting data from SEMCOR")
    semcor_reader = SemCorSelector()
    semcor_reader.get_word_data(word, pos)
    print("Getting sentences for relevant senses")
    sel_senses = semcor_reader.select_senses(min_senses)
    sentences, trees, sense_indices = semcor_reader.get_selected_sense_sents(sel_senses)
    tree_labels = get_sense_labels(sense_indices, sel_senses)
    print("Generating BERT embeddings")
    activations, attn_tensors, tokenized_texts = get_activations_attns(word, pos, trees, model)
    embeddings = process_raw_embeddings(activations, 4, sum_layers)
    avgd_attns_for_layers = process_raw_attentions(attn_tensors, tokenized_texts)
    result_dict = {'lemma': semcor_reader.curr_word, 'embeddings': embeddings, 'sense_indices': sense_indices, 
    'original_sentences': sentences, 'sense_names': sel_senses, 'sense_labels': tree_labels, "all_activations": activations, "attns": avgd_attns_for_layers}
    if savefile:
        #Convert Pytorch/Numpy arrays to Python lists
        json_dict = result_dict.copy()
        json_dict['embeddings'] = [v.tolist() for v in result_dict['embeddings']]
        json_dict['sense_names'] = [str(s) for s in sel_senses]
        json_dict['all_activations'] = [m.tolist() for m in result_dict['all_activations']]
        json_dict['attns'] = [{k: d[k].tolist() for k in d} for d in result_dict['attns']]
        write_json(json_dict, word, pos, dir_name)
    return result_dict

def run_pipeline_df(word, pos, df, model, savefile = False):
    """
    Input:
    word- string, name of lemma
    pos- string, part of speech (n, v, s)
    model- pre-trained BERT model from Hugging Face
    savefile- boolean, says whether we want to save the embeddings
    df- a Pandas dataframe that has the following schema: one row per lemma, columns contain lemma, part of speech ('pos'), WordNet Sense ('wn_sense'), 
    and sentence the type is used in ('sentence'), column names are named in parentheses in the docstrings

    Output:
    result_dict = {
        'lemma': word.pos
        'embeddings': list of 768 x 1 PyTorch tensors representing vectorized representations of the lemma in a sentence in SEMCOR
        'original_sentences': List of strings representing sentences where each embedding came from
        'sense_labels': List of strings representing WordNet senses (name.pos.number), that specifies the sense of the lemma used in each sentence 
    }

    """
    #First designed to work with the Google Word Sense dataset, pos is allcaps 
    word_df = df[(df['lemma'] == word) & (df['pos'] == pos)]
    embeddings = []
    sense_names = []
    sentences = []
    df = df.reset_index()
    for i in df.index:
        row = df.iloc[i]
        activations = get_embeddings_attns(preprocess(row['sentence'], row['word']), model)
        embeddings.append(process_raw_embeddings([activations], 4, sum_layers)[0])
        sense_names.append(row['wn_sense'])
        sentences.append(row['sentence'])
    #All we need to do is save the embeddings somewhere, we have the information we store for the SEMCOR words in our corpus
    result_dict = {'lemma': word + '.' + pos, 'embeddings': embeddings, 'original_sentences': sentences, 'sense_labels': sense_names}
    if savefile:
        json_dict = result_dict.copy()
        json_dict['embeddings'] = [v.tolist() for v in result_dict['embeddings']]
        write_json(json_dict, word, pos, 'masc')
    return result_dict


def load_data(word, pos, corpus_dir):
    """
    Opens JSON of embeddings that lives at filepath ../data/pipeline_results/[corpus_dir]/[word]_[pos].json
    """
    fname = word + '_' + pos + '.json'
    fpath = os.path.join('..', 'data', 'pipeline_results', corpus_dir, fname)
    with open(fpath, 'r') as path:
        results = json.load(path)
    return results        

def save_embeds_with_wts(embeddings, sense_labels, weights, lemma):
    """
    Inputs:
    embeddings- list of 768 dimensional vectors for a lemma (word.pos)
    sense_labels- List of strings representing WordNet senses (name.pos.number), that specifies the sense of the lemma represented by each BERT vector
    weights- list of lists of 768 elements, each represents the weights used by the logistic regression probe to discriminate between a pair of senses
    lemma- string of word and part of speech

    Outputs:
    Saves values of BERT embeddings at the indices of nonzero weights and the sense_labels at ../data/pipeline_results/select_weights/word_pos.json
    """
    nonzero_weight_indices = np.unique(np.concatenate([np.nonzero(weights[i])[0] for i in range(len(weights))]))
    impt_weight_values = [e[nonzero_weight_indices].tolist() for e in embeddings]
    print(lemma, "Proportion of Weights that are Nonzero", len(impt_weight_values) / 768)
    json_dict = {'embeddings': impt_weight_values, 'sense_labels': sense_labels.tolist()}
    word, pos = lemma.split('.')
    write_json(json_dict, word, pos, 'select_weights')