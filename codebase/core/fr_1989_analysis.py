from . import semcor_bert_pipeline as bert_client
import numpy as np
import pandas as pd

def sentence_pair_comparison(c1, c2, end_index):
    #c1 has no "padding," c2 does
    """
    c1 and c2 are two dataframes of stimuli from the experiment(either 1 or 2)
    bias refers to if the sense distribution is unbiased/biased (0 or 1)
    d_s denotes if the sense is dominant/subordinate 
    early_late (Experiment 1) tells us if the disambiguating token is right after the target or not
    before_after (Experiment 2) tell us if the disambiguating token is right after
    end_index is 1 + the amount of stimuli that were given for each bias condition (biased/unbiased)
    target_token: ambiguous token in the sentence
    disambig_token: Experiment 1- token that specifies the sense of target_token
    Expriment 2- where disambiguating info starts(usually neutral words like because/after or commas)
    
    Outputs lists of tuples, one for each dataframe
    Tuples are of format (attention: {layer_num(integer from 0-11): 
        {"attn_vector": 
        vector that corresponds to the attention the target token gives to all other tokens for layer (array),
        "tokens": text as tokenized by BERT, getting rid of SEP/CLS and punctuation (list),
        d_idx: index of disambiguating token (NOTE: the token could have been a comma for Expt. 2,
        we want the sentence after the comma), target_idx: index of target token}...}, target_token_embedding)
    
    """
    model = bert_client.initialize_model()
    data_1 = []
    data_2 = []
    for bias in [0, 1]:
        for stimulus_index in np.arange(1, end_index):
            row_1 = c1[(c1['bias'] == bias) & (c1['stimulus_index'] == stimulus_index)].T.squeeze()
            row_2 = c2[(c2['bias'] == bias) & (c2['stimulus_index'] == stimulus_index)].T.squeeze()
            s1, d_tok_1 = row_1['sentence'], row_1['dt_stem']
            s2, d_tok_2 = row_2['sentence'], row_2['dt_stem']
            assert row_1['target_token'] == row_2['target_token']
            target = row_1['target_token']
            data_1.append(target_attns(s1, target, d_tok_1, model))
            data_2.append(target_attns(s2, target, d_tok_2, model))
    return data_1, data_2

def top_layers(early_attn, late_attn):
    """
    Input: 2 attention dictionaries (12 keys eacy)
    attention: {layer_num(integer): 
        {"attn_vector": 
        vector that corresponds to the attention the target token gives to all other tokens for layer (array),
        "tokens": text as tokenized by BERT, getting rid of SEP/CLS and punctuation (list),
        d_idx: index of disambiguating token (NOTE: the token could have been a comma for Expt. 2,
        we want the sentence after the comma), target_idx: index of target token}...}
        
    Output: List of Sets consisting of the number of layers where the disambiguating token has the most attention
    for both stimuli
    
    """
    common_layers = []
    for early_dict, late_dict in zip(early_attn, late_attn):
        top_token_e = set()
        top_token_l = set()
        for layer in early_dict.keys():
            early_attn = np.array(early_dict[layer]['attn_vector'])
            late_attn = np.array(late_dict[layer]['attn_vector'])
            if np.argsort(-early_attn)[0] == early_dict[layer]['d_idx']:
                top_token_e.add(layer)
            if np.argsort(-late_attn)[0] == late_dict[layer]['d_idx']:
                top_token_l.add(layer)
        common_layers.append(top_token_e.intersection(top_token_l))
    return common_layers

def target_attns(sentence, target_token, disambig_token, model):
    """
    Inputs: sentence- sentence from experiment
    target_token- token we are getting BERT embeddings for
    disambig_token- token that reveals information about target_token's sense
    model- pretrained BERT model
    
    Output:
    List of ranks of disambig_token in the attention vector for target_token. 
    Indices in the list correspond to layers.
    """
    indexed_tokens, tokenized_text = bert_client.preprocess(sentence, target_token)
    target_activations, attns = bert_client.get_model_output(indexed_tokens, model)
    attn_dict = bert_client.process_raw_attentions([attns], [tokenized_text])[0]
    d_tok_idx = tokenized_text[0][1:-2].index(disambig_token) #index of the disambiguating token, removing SEP/CLS
    output_dict = {}
    target_embeddings = sum_layers(target_activations, -4)
    for k in attn_dict.keys():
        attn_vector = attn_dict[k][1:-2].tolist()
        text_tokens = tokenized_text[0][1:-2]
        if "," in text_tokens:
            comma_index = text_tokens.index(",")
            text_tokens.pop(comma_index)
            attn_vector.pop(comma_index)
            
        target_idx = text_tokens.index(target_token)
 
        output_dict[k] = {"attn_vector": attn_vector, "tokens": text_tokens,
                          "d_idx": d_tok_idx, "target_idx": target_idx}
    return output_dict, target_embeddings
    #return np.array([np.argwhere(np.argsort(-attn_dict[k]) == d_tok_idx)[0][0] for k in attn_dict])
                                                                                                     
#utils

def avg_pairwise_cosine_sim(l1, l2):
    #l1 and l2 are lists of embeddings
    assert len(l1) == len(l2)
    n = len(l1)
    return np.mean([cosine_sim(l1[i], l2[i]) for i in range(n)])

first_item = lambda l: [i[0] for i in l]
second_item = lambda l: [i[1] for i in l]
