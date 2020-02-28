import os
import sys
import argparse
import pandas as pd
from semcor_bert_pipeline import *
from clustering import *
from fpdf import FPDF


def run_clustering(word, pos, model, gmm_dr):
    #either PCA or TSNE
    print(word, pos)
    fname = word + '_' + pos
    word_results = run_pipeline(word, pos, model, min_senses = 10, savefile = True)
    print("Generating Plots")
    tsne_results = plot_embeddings(word_results['embeddings'], word_results['sense_indices'], word_results['sense_names'], word_results['lemma'], savefile = True)
    color_dict, label_dict = create_dendrogram_colors(word_results['sense_names'])
    plot_dendrogram(word_results, color_dict, label_dict, savefile = True)
    print("Running GMM Simulations")
    if gmm_dr == 'PCA':
        raw_json = plot_gmm_rand_indices(word_results, range(2, 30), save_json = True)
    if gmm_dr == 'TSNE':
        return tsne_rand(word_results)
    else:
        tsne_rand_indices = tsne_rand(word_results)
        rand = plot_gmm_rand_indices(word_results, range(2, 4))
        pca_rand_indices = process_pca(rand, range(2, 4), word, pos)
        return tsne_rand_indices, pca_rand_indices

def convert_imgs_to_pdf(name, pos):
    pdf = FPDF('L') 
    pdf.set_auto_page_break(0)
    img_fpath = os.path.join('data', 'clustering_results', name + '_' + pos)
    img_files = [i for i in os.listdir(img_fpath) if i.endswith('png')]

    for f in img_files:
        pdf.add_page()
        path = os.path.join(img_fpath, f)
        pdf.image(path, 0, 0)
    print("Saving as PDF")
    pdf.output("../results/clustering_images/" + name + "_" + pos + ".pdf", "F")

def run_all_sparse():
    sparse_senses = pd.read_csv('data/semcor_sparsity.csv')
    #be, have, see
    skipped_words = []
    for i in range(len(sparse_senses.index)):
        row = sparse_senses.iloc[i]
        word, pos = row['word'], row['pos']
        dir_name = os.path.join("data", 'clustering_results', word + '_' + pos)
        os.system('mkdir ' + dir_name)
        try: 
            run_clustering(word, pos, model, 'GMM')
            convert_imgs_to_pdf(word, pos)
        except:
            skipped_words.append(word + '.' + pos)
            continue
    write_to_file(skipped_words, os.path.join('data', 'skipped_sparse_words.txt'))

def run_tsne_entropy():
    #TODO SPLIT THIS UP
    sparse_senses = pd.read_csv('data/semcor_entropy.csv')
    completed_files = os.listdir(os.path.join('data', 'pipeline_results', 'sparse'))
    all_rand_tsne = []
    all_rand_gmm = []
    failed_words = []
    index_range = range(len(sparse_senses.index))
    tsne_result_df = pd.read_csv("data/tsne_rand_indices.csv")
    pca_result_df = pd.read_csv('data/gmm_rand_indices.csv')
    if len(tsne_result_df.index) and len(pca_result_df.index):
        all_rand_tsne = tsne_result_df.to_dict().values()
        all_rand_gmm = pca_result_df.to_dict().values()
        index_dict = tsne_result_df.to_dict('index') #Both dataframes should have the same words
        last_row = index_dict[len(index_dict.keys()) - 1]
        last_word, last_pos = last_row['Lemma'].split('.')[0], last_row['Lemma'].split('.')[1]
        start_index = sparse_senses[(sparse_senses['word'] == last_word) & (sparse_senses['pos'] == last_pos)].index[0] + 1
        end_index = len(sparse_senses.index)
        index_range = range(start_index, end_index)
    for i in index_range:
        row = sparse_senses.iloc[i]
        word, pos = row['word'], row['pos']
        json_name = word + "_" + pos + '.json'
        print("Processing", word, pos)
        if json_name in completed_files:
            print("Found logged data")
            with open(os.path.join('data', 'pipeline_results', 'sparse', json_name), 'r') as fpath:
                word_results = json.load(fpath)
            try:
                print("Running GMM+TSNE")
                tsne_rand_indices = tsne_rand(word_results)
                print("Running GMM+PCA")
                gmm_rand_indices = process_pca(plot_gmm_rand_indices(word_results, range(2, 4)), range(2, 4), word, pos)
                all_rand_tsne += tsne_rand_indices
                all_rand_gmm += gmm_rand_indices
            except:
                failed_words.append(word + '.' + pos)
        else:
            try:
                dir_name = os.path.join("data", 'clustering_results', word + '_' + pos)
                os.system('mkdir ' + dir_name)
                tsne_rand_indices, gmm_rand_indices = run_clustering(word, pos, model, 'both')
                all_rand_tsne += tsne_rand_indices
                all_rand_gmm += gmm_rand_indices
            except:
                failed_words.append(word + '.' + pos)
    
        if i % 5 == 0:
            print("Saving results to disk")
            pd.DataFrame(all_rand_tsne).to_csv('data/tsne_rand_indices.csv', index = False)
            pd.DataFrame(all_rand_gmm).to_csv('data/gmm_rand_indices.csv', index = False)
        
    write_to_file(failed_words, os.path.join('data', 'skipped_sparse_words.txt'))

def run_test():
    #For testing purposes, default to word table
    word, pos = 'table', 'n'
    dir_name = os.path.join("data", 'clustering_results', word + '_' + pos)
    os.system('mkdir ' + dir_name)
    run_clustering(word, pos, model, 'both')
    #convert_imgs_to_pdf(word, pos)

def run_sparse_pca():
    result_path = os.path.join('data', 'pipeline_results')
    jsons = [i for i in os.listdir(result_path) if i not in ['bat_n.json', 'table_n.json']]
    all_sparse_embeds = []
    for js in jsons:
        json_path = os.path.join(result_path, js)
        with open(json_path, 'r') as fpath:
            word_results = json.loads(fpath)
        all_sparse_embeds += word_results['embeddings']
    plot_pca_ev(range(2, 40), all_sparse_embeds, "Sparse Words in SEMCOR")
        

def write_to_file(lst, fpath):
    with open(fpath, 'w') as f:
        for i in lst:
            f.write(i + '\n')

def check_for_embedding_data(word, pos):
    fname = word + '_' + pos + '.json'
    if fname in os.listdir(os.path.join('data', 'pipeline_results')):
        return 1
    else:
        return 0

if __name__ == '__main__':
    model = initialize_model()
    if sys.argv[1] == '--test':
        run_test()
    elif sys.argv[1] == '--pca':
        run_all_sparse()
    elif sys.argv[1] == '--tsne_entropy':
        run_tsne_entropy()
        
