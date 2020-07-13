import itertools
import pandas as pd
from core.metrics import cosine_sim
from core.semcor_bert_pipeline import *

#Modeling functions
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

def masking_embeds_preds(df):
    """
    df- columns for s1, s2- sentence 1 and 2, and target token
    
    Outputs list of dictionaries with mask embeddings, predicted tokens, and predicted probabilities
    """
    mask_model = initialize_masking_lm()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mask_results = []
    for i in range(len(df.index)):
        row = df.iloc[i]
        s1, s2, target_token = row['s1'], row['s2'], row['target']
        indexed_mask, _ = preprocess(s1, target_token, s2 = s2, masking = True)
        indexed_target, _ = preprocess(s1, target_token, s2 = s2)
        mask_embed, preds = mask_predictions(indexed_mask, mask_model, tokenizer)
        target_activations, _ = get_model_output(indexed_target, bert_model)
        target_embeds = sum_layers(target_activations, -4)
        mask_results.append({'mask_embed': mask_embed, 'predicted_tokens': preds[1], 'predicted_probs': preds[0]})
    return mask_results

#Analytics
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

def generate_sense_predictions(df, mask_results):
    """
    Input:
    df- experimental stimuli
    mask_results- list of dictionaries with mask embeddings and predictions (we only use embeddings)
    
    all_data- Dataframe of experimental stimuli with 1-nearest neighbors and centroid-based sense predictions
    for each sense
    """
    #Two senses
    reg_cases = df[df['target'].isin(both_sense_types[:-4])]
    e_preds = []
    c_preds = []
    #Polysemes/homonyms where we have 2 senses
    for w in reg_cases['target'].unique():
        indices = df[df['target'] == w].index
        senses = df[df['target'] == w]['wn_sense'].unique()
        mask_embeddings = [mask_results[i]['mask_embed'] for i in indices]
        if w == 'herbs':
            semcor_word = 'herb'
        else:
            semcor_word = w
        token_data = load_data(semcor_word, 'n', 'semcor')
        semcor_embeddings = np.array(token_data['embeddings'])
        num_embeddings = len(semcor_embeddings)
        s1_embeds = semcor_embeddings[[i for i in range(num_embeddings) if token_data['sense_labels'][i] == senses[0]]]
        s2_embeds = semcor_embeddings[[i for i in range(num_embeddings) if token_data['sense_labels'][i] == senses[1]]]
        s1_centroid, s2_centroid = centroid(s1_embeds), centroid(s2_embeds)
        e_preds += [nearest_neighbor(e, s1_embeds, s2_embeds, senses) for e in mask_embeddings]
        c_preds += [centroid_pred(e, s1_centroid, s2_centroid, senses) for e in mask_embeddings]

    #Homophones
    homophones = [('son', 'sun'), ('night', 'knight')]
    for p in homophones:
        t1, t2 = load_data(p[0], 'n', 'semcor'), load_data(p[1], 'n', 'semcor')
        s1, s2 = df[df['target'] == p[0]]['wn_sense'].values[0], df[df['target'] == p[1]]['wn_sense'].values[0]
        mask_indices = list(df[df['target'] == p[0]].index) + list(df[df['target'] == p[1]].index)
        mask_embeddings = [mask_results[i]['mask_embed'] for i in mask_indices]
        s1_embeds = np.array(t1['embeddings'])[[i for i in range(len(t1['embeddings'])) if t1['sense_labels'][i] \
                                                == s1]]
        s2_embeds = np.array(t2['embeddings'])[[i for i in range(len(t2['embeddings'])) if t2['sense_labels'][i] \
                                                == s2]]
        e_preds += [nearest_neighbor(e, s1_embeds, s2_embeds, [s1, s2]) for e in mask_embeddings]
        s1_centroid, s2_centroid = centroid(s1_embeds), centroid(s2_embeds)
        c_preds += [centroid_pred(e, s1_centroid, s2_centroid, [s1, s2]) for e in mask_embeddings]
    two_senses = df[df['target'].isin(both_sense_types)]
    two_senses['nn_preds'] = e_preds_mask
    two_senses['centroid_preds'] = c_preds_mask

    #Types where we only have one sense in SEMCOR
    one_sense = df[df['target'].isin(one_sense_types)]
    one_sense_nn = []
    one_sense_centroid = []
    for w, s in zip(one_sense_types, available_sense):
        indices = df[df['target'] == w].index
        senses = df[df['target'] == w]['wn_sense'].unique()
        mask_embeddings = [mask_results[i]['mask_embed'] for i in indices]
        target_embeddings = [embed_results[i] for i in indices]
        if w == 'cards':
            semcor_word = 'card'
        else:
            semcor_word = word
        token_data = load_data(w, 'n', 'semcor')
        semcor_embeddings = np.array(token_data['embeddings'])
        num_embeddings = len(semcor_embeddings)
        sense_embeds = semcor_embeddings[[i for i in range(num_embeddings) if token_data['sense_labels'][i] == s]]
        other = [sen for sen in senses if sen != s][0]

        if w == 'glasses': #Other sense is plural form of glass.n
            glass_embeds = load_data('glass', 'n', 'semcor')
            item_embeds = np.array(glass_embeds['embeddings'])[[i for i in range(len(glass_embeds['embeddings'])) \
                                                  if glass_embeds['sense_labels'][i] == other]]
            one_sense_nn += [nearest_neighbor(e, sense_embeds, item_embeds, [s1, s2]) for e in mask_embeddings]
            s1_centroid, s2_centroid = centroid(sense_embeds), centroid(item_embeds)
            one_sense_centroid += [centroid_pred(e, s1_centroid, s2_centroid, [s, other]) for e in mask_embeddings]

        else:
            one_sense_nn += one_sense_preds(mask_embeddings, semcor_embeddings, s, other, 'nn')
            one_sense_centroid += one_sense_preds(mask_embeddings, semcor_embeddings, s, other, 'nn')

    one_sense['nn_preds'] = one_sense_nn
    one_sense['centroid_preds'] = one_sense_centroid
    all_data = pd.concat([two_senses, one_sense])
    return all_data

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
def check_semcor_data(word_pos_pairs, df):
    """
    word_pos_pairs has one row for each type in df(the stimulus dataframe)
    
    This function loads the data for each type and checks if there are data for 1 or 2 sensese for the type in SEMCOR.
    both_sense_types contains the words with that have both senses tested in the experiment in SEMCOR, and one_sense_types is a list of the types that
    only had one sense with SEMCOR data
    """
    both_sense_types = []
    one_sense_types = []
    for w in word_pos_pairs.iterrows():
        w = w[1]
        word, pos = w['word'], w['pos']
        if word == 'cards':
            semcor_word = 'card'
        elif word == 'herbs':
            semcor_word = 'herb'
        else:
            semcor_word = word
        try:
            type_data = load_data(semcor_word, pos, 'semcor')
            word_senses = df[df['target'] == word]['wn_sense'].unique()
            contains_both_senses = lambda l1, l2: all([s in l2 for s in l1])
            if contains_both_senses(word_senses, type_data['sense_labels']):
                both_sense_types.append(word)
            common_senses = set(word_senses).intersection(set(type_data['sense_labels']))
            if len(common_senses) == 1 and word not in ['son', 'sun', 'night', 'knight']:
                one_sense_types.append(word)
                available_sense.append(common_senses.pop())
                
        except:
            pass
    return both_sense_types, one_sense_types

def nearest_neighbor(query, s1, s2, sense_names):
    """
    Returns the sense that better corresponds to the query embedding with a nearest neighbors approach.
    s1 and s2 are lists of embeddings in SEMCOR for two different senses, sense_names[0] is the sense of 
    the s1 vectors, and sense_names[1] is the sense of s2
    """
    s1_sims = [cosine_sim(query, e) for e in s1]
    s2_sims = [cosine_sim(query, e) for e in s2]
    if max(s1_sims) > max(s2_sims):
        return sense_names[0]
    if max(s1_sims) < max(s2_sims):
        return sense_names[1]

def centroid_pred(query, c1, c2, sense_names):
    """
    Returns the sense that better corresponds to the query embedding. c1 and c2 are centroids of the
    SEMCOR embeddings for two different senses, sense_names[0] is the sense of the c1 centroid,
    and sense_names[1] is the sense of c2
    """

    c1_sim = cosine_sim(query, c1)
    c2_sim = cosine_sim(query, c2)
    if c1_sim > c2_sim:
        return sense_names[0]
    if c1_sim < c2_sim:
        return sense_names[1]

def one_sense_preds(query_embeds, sense_embeds, data_sense, other_sense, method):
    """
    query_embeds- list of 4 embeddings we want to predict the sense for
    sense_embeds- embeddings for the sense we have data in SEMCOR for (data_sense)
    other_sense- sense that we don't have data for (str)
    method is either "nn" or "centroid"
    
    Returns predicted senses for each item in query_embeds with either 1-nearest-neighbors or centroid method.
    This is for cases where we only have SEMCOR data for one sense
    """
    if method == "nn":
        cos_sims = [max([cosine_sim(q, s) for s in sense_embeds]) for q in query_embeds]
    if method == 'centroid':
        c = centroid(sense_embeds)
        cos_sims = [cosine_sim(q, c) for q in query_embeds]
    sense_preds = ["" for i in range(4)]
    sorted_indices = np.argsort(cos_sims)
    least_sim, most_sim = sorted_indices[:2], sorted_indices[2:]
    for i in range(len(sense_preds)):
        if i in least_sim:
            sense_preds[i] = other_sense
        if i in most_sim:
            sense_preds[i] = data_sense
    return sense_preds

def accuracy_by_predicate(df, predicate, value, prediction_type):
    """
    Takes in dataframe with sense predictions (nn_preds and centroid_preds) and returns accuracy by predicate-
    rel_type(homonymy/polysemy) sentence_ctx(previous/current)
    """
    df = df[df[predicate] == value]
    return sum(df[prediction_type] == df['wn_sense']) / len(df.index)

def accuracy_table(all_data, predicate):
    """
    all_data has sense predictions for each stimulus, this creates a table with accuracy for centroid/nearest
    neighbor based predictions for predicate(string). For this dataset, good predicates would be rel_type (polysemy/
    homonymy) and sentence_ctx(disambiguating context in current/previous sentence)
    """
    return pd.DataFrame({predicate: all_data[predicate].unique(),
                         "centroid_acc": [accuracy_by_predicate(all_data, predicate,
                                                                 p, 'centroid_preds') for p in df[predicate].unique()],
             "nn_acc": [accuracy_by_predicate(all_data, predicate, p, 'nn_preds') for p in df[predicate].unique()]})
            