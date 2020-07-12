import itertools
import pandas as pd
from core.metrics import cosine_sim
from core.semcor_bert_pipeline import *

def extract_embeddings_attns(df):
    """
    Input: ../data/rabagliati_13_stimuli.csv
    Output:
    bert_data - list of dictionaries, one for each row of the dataframe (stimulus)
    {"attns": {layer(int from 0-11): attention vector (array),
    "target_embed": Pytorch array of embeddings for the target token, "target_name": target_token name,
    "tokenized_sents": text from BertTokenizer, "tokenized_dis": tokenized disambiguating context
                                 "sent_ctx": sent_ctx}
    """
    bert_data = []
    for i in np.arange(0, len(df.index), 4):
        type_stimuli = df.iloc[i:i + 4]
        for sent_ctx in ['curr', 'prev']:
            curr_prev_ctx = type_stimuli[type_stimuli['sentence_ctx'] == sent_ctx]
            for row in curr_prev_ctx.iterrows():
                row = row[1]
                s1, s2, target_token = row['s1'], row['s2'], row['target']
                indexed_tokens, tokenized_text = preprocess(s1, target_token, s2 = s2)
                target_activations, attns = get_model_output(indexed_tokens, model)
                attn_dict = process_raw_attentions([attns], [tokenized_text])[0]
                target_embeddings = sum_layers(target_activations, -4)
                bert_data.append({"attns": attn_dict, "target_embed": target_embeddings, "target_name": target_token,
                                "tokenized_sents": tokenized_text[0], "tokenized_dis": row['tokenized_dis_ctx'],
                                "sent_ctx": sent_ctx})
    return bert_data


def hp_cs(bert_data):
    """
    Inputs:
    bert_data - list of dictionaries, one for each row of the dataframe (stimulus)
    {"attns": {layer(int from 0-11): attention vector (array),
    "target_embed": Pytorch array of embeddings for the target token, "target_name": target_token name,
    "tokenized_sents": text from BertTokenizer, "tokenized_dis": tokenized disambiguating context
                                 "sent_ctx": sent_ctx}
    
    Outputs:
    Dataframe with columns- hp_data: one of reg_pol, irreg_pol, hom
    cosine_sim- cosine similarity between a pair of senses with this relationship
    """
    find_sim = lambda l, q: [t[1] for t in l if t[0] == q] 
    hp_data = []
    for i in np.arange(0, len(bert_data), 4):
        word_dicts = bert_data[i:i + 4]
        word_embeddings = [d['target_embed'] for d in word_dicts]
        cs_pairs = []
        for j, k in itertools.combinations(np.arange(4), 2):
            if (j, k) not in [(0, 2), (1, 3)]:
                cs_pairs.append(((j, k), cosine_sim(word_embeddings[j], word_embeddings[k])))
        if (i / 4) < 8 and (i / 4) >= 0:
            hp_data += [{'rel_type': 'reg_pol', 'cosine_sim': i[1]} for i in cs_pairs]
        elif (i / 4) < 16 and (i / 4) >= 8:
            hp_data += [{'rel_type': 'irreg_pol', 'cosine_sim': i[1]} for i in cs_pairs]
        else:
            hp_data += [{'rel_type': 'hom', 'cosine_sim': i[1]} for i in cs_pairs]
    hp_data = pd.DataFrame(hp_data)
    return hp_data

def avg_attn(bert_data):
    """
    Inputs:
    bert_data - list of dictionaries, one for each row of the dataframe (stimulus)
    {"attns": {layer(int from 0-11): attention vector (array),
    "target_embed": Pytorch array of embeddings for the target token, "target_name": target_token name,
    "tokenized_sents": text from BertTokenizer, "tokenized_dis": tokenized disambiguating context
                                 "sent_ctx": sent_ctx}
    Outputs:
    Dataframe with columns layer- int 0 through 11, avg_attn- average attention for tokens corresponding to tokens
    column, tokens- one of [prev_uninformative, prev_disambig, curr_uninformative, curr_disambig]
    ctx_sentence- one of [prev, curr]
    
    """

    avg_attn_split = []
    for d in bert_data:
        sentence = d['tokenized_sents']
        target = d['target_name']
        sep_index = sentence.index("[SEP]")
        prev_sent, prev_indices = np.array(sentence[:sep_index]), lex_indices(sentence[:sep_index], target)
        curr_sent, curr_indices = np.array(sentence[sep_index + 1:]), lex_indices(sentence[sep_index + 1:] , target)
        prev_sent, curr_sent = prev_sent[prev_indices], curr_sent[curr_indices]

        if d['sent_ctx'] == 'curr':
            dis_indices = find_sub_list(d['tokenized_dis'], curr_sent.tolist())
        else:
            dis_indices = find_sub_list(d['tokenized_dis'], prev_sent.tolist())

        if dis_indices is not None:
            dis_start = dis_indices[0]
            dis_end = dis_indices[1] + 1
            for layer in d['attns']:
                attn_lst = d['attns'][layer]
                prev_attns = attn_lst[prev_indices]
                curr_attns = attn_lst[sep_index + 1:][curr_indices]
                if d['sent_ctx'] == 'curr':
                    avg_attn_split.append({'layer': layer, "avg_attn": np.mean(prev_attns),
                                     'tokens': 'prev_uninformative', 'ctx_sentence': d['sent_ctx']})
                    avg_attn_split.append({'layer': layer,
                                     "avg_attn": np.mean(curr_attns[:dis_start].tolist() + curr_attns[dis_end:].tolist()),
                         'tokens': 'curr_uninformative', 'ctx_sentence': d['sent_ctx']})
                    avg_attn_split.append({'layer': layer,
                                 "avg_attn": np.mean(curr_attns[dis_start:dis_end]),
                     'tokens': 'curr_disambig', 'ctx_sentence': d['sent_ctx']})
                if d['sent_ctx'] == 'prev':
                    avg_attn_split.append({'layer': layer, "avg_attn": np.mean(prev_attns[dis_start:dis_end]),
                     'tokens': 'prev_disambig', 'ctx_sentence': d['sent_ctx']})
                    avg_attn_split.append({'layer': layer,
                                     "avg_attn": np.mean(prev_attns[:dis_start].tolist() + prev_attns[dis_end:].tolist()),
                         'tokens': 'prev_uninformative', 'ctx_sentence': d['sent_ctx']})
                    avg_attn_split.append({'layer': layer, "avg_attn": np.mean(curr_attns),
                     'tokens': 'curr_uninformative', 'ctx_sentence': d['sent_ctx']})

    avg_attn_split = pd.DataFrame(avg_attn_split)
    return avg_attn_split

def total_attn_location(bert_data):
    """
    Inputs:
    bert_data - list of dictionaries, one for each row of the dataframe (stimulus)
    {"attns": {layer(int from 0-11): attention vector (array),
    "target_embed": Pytorch array of embeddings for the target token, "target_name": target_token name,
    "tokenized_sents": text from BertTokenizer, "tokenized_dis": tokenized disambiguating context
                                 "sent_ctx": sent_ctx}
    
    Output:
    Dataframe with columns- layer (int from 0 to 11), prior_attn- total attention on tokens before target
    target_attn- total attention on tokens after target, sent_ctx- [prev or curr]
    """
    curr_prev_sent = []
    for d in bert_data:
        sep_index = d['tokenized_sents'].index("[SEP]")
        for l in d['attns']:
            sum_prev = np.sum(d['attns'][l][1:sep_index - 1])
            sum_curr = np.sum(d['attns'][l][sep_index + 1:-2])
            curr_prev_sent.append({"layer": l, 'prior_attn': sum_prev,
                                   'target_attn': sum_curr, "sent_ctx": d['sent_ctx']})
        curr_prev_sent = pd.DataFrame(curr_prev_sent)

        return curr_prev_sent

def disambig_ranks_attn(bert_data):
    """
    Inputs:
    bert_data - list of dictionaries, one for each row of the dataframe (stimulus)
    {"attns": {layer(int from 0-11): attention vector (array),
    "target_embed": Pytorch array of embeddings for the target token, "target_name": target_token name,
    "tokenized_sents": text from BertTokenizer, "tokenized_dis": tokenized disambiguating context
                                 "sent_ctx": sent_ctx}
    
    Outputs:
    attn_ranks: Dataframe with columns: layer- (int 0-11), normed_ranks- float from 0 to 1, list of ranks of disambig. tokens
    divided by sentence length, dis_location ['prev' or 'curr'], specifies where disambiguating context is
    
    attn_values: Dataframe with columns: layer- (int 0-11), attn- attention weights for disambiguating tokens,
    dis_location ['prev' or 'curr'], specifies where disambiguating context is
                                 
    """
    prev_attn = []
    curr_attn = []
    prev_layers = []
    curr_layers = []
    failures = []
    prev_attn_ranks = []
    curr_attn_ranks = []
    prev_rank_layers = []
    curr_rank_layers = []
    for d in bert_data:
        lex_indices = [i for i in range(len(d['tokenized_sents'])) if d['tokenized_sents'][i] not in ["[CLS]", "[SEP]", ".", ","]]
        lex_tokens = np.array(d['tokenized_sents'])[lex_indices]
        dis_indices = find_sub_list(d['tokenized_dis'], lex_tokens.tolist())
        if dis_indices is not None:
            dis_start = dis_indices[0]
            dis_end = dis_indices[1] + 1
            target_indices = np.arange(dis_start, dis_end)
            for layer in d['attns']:
                a = d['attns'][layer][lex_indices]
                summed_attns = np.sum(a[dis_start:dis_end])
                attn_ranks = [i / len(lex_tokens) for i in np.arange(len(a)) if np.argsort(a)[i] in target_indices]
                if d['sent_ctx'] == 'prev':
                    prev_attn.append(summed_attns)
                    prev_attn_ranks += attn_ranks
                    prev_rank_layers += [layer] * len(attn_ranks)
                    prev_layers.append(layer)
                if d['sent_ctx'] == 'curr':
                    curr_attn.append(summed_attns)
                    curr_attn_ranks += attn_ranks
                    curr_rank_layers += [layer] * len(attn_ranks)
                    curr_layers.append(layer)
        else:
            failures.append((d['tokenized_dis'], d['tokenized_sents']))
    attn_ranks = pd.DataFrame({"layer": prev_rank_layers + curr_rank_layers,
                               "normed_ranks": prev_attn_ranks + curr_attn_ranks,
              "dis_location": (["prev_sentence"] * len(prev_rank_layers)) + (['curr_sentence'] * len(curr_rank_layers))})
    attn_values = pd.DataFrame({"layer": prev_layers + curr_layers, "attn": prev_attn + curr_attn,
              "dis_location": (["prev_sentence"] * len(prev_attn)) + (['curr_sentence'] * len(curr_attn))})
    return attn_ranks, attn_values

#Utility functions
lex_indices = lambda sent, targ: [i for i in range(len(sent)) if sent[i] not in ["[CLS]", "[SEP]", targ,
                                                                                      ".", ","]]
#Get indices for sentence that are not CLS, SEP, punctuation, or target

def find_sub_list(sl,l):
    """
    Returns indices of a list(l) contained in a sublist (sl)
    """
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1


