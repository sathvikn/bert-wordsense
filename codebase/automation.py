import os
import sys
import argparse
import pandas as pd
from semcor_bert_pipeline import *
from clustering import *
from fpdf import FPDF


def run_clustering(word, pos, model):
    print(word, pos)
    fname = word + '_' + pos
    word_results = run_pipeline(word, pos, model, min_senses = 10, savefile = True)
    print("Generating Plots")
    tsne_results = plot_embeddings(word_results['embeddings'], word_results['sense_indices'], word_results['sense_names'], word_results['lemma'], savefile = True)
    color_dict, label_dict = create_dendrogram_colors(word_results['sense_names'])
    plot_dendrogram(word_results, color_dict, label_dict, savefile = True)
    print("Running GMM Simulations")
    raw_json = plot_gmm_rand_indices(word_results, range(2, 30), save_json = True)

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
            run_clustering(word, pos, model)
            convert_imgs_to_pdf(word, pos)
        except:
            skipped_words.append(word + '.' + pos)
            continue
    write_to_file(skipped_words, os.path.join('data', 'skipped_sparse_words.txt'))

def run_test():
    #For testing purposes, default to word table
    word, pos = 'table', 'n'
    dir_name = os.path.join("data", 'clustering_results', word + '_' + pos)
    os.system('mkdir ' + dir_name)
    run_clustering(word, pos, model)
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
        f.write(lst)

def check_for_embedding_data(word, pos):
    fname = word + '_' + pos + '.json'
    if fname in os.listdir(os.path.join('data', 'pipeline_results')):
        return 1
    else:
        return 0

def run_gmm_existing():
    for f in os.listdir(os.path.join('data', 'pipeline_results', 'sparse')):
        datapath = os.path.join('data', 'pipeline_results', 'sparse' f)
        print(datapath)
        with open(datapath, 'r') as fp:
            word_results = json.load(fp)
        raw_json = plot_gmm_rand_indices(word_results, range(2, 30), save_json = True)

if __name__ == '__main__':
    model = initialize_model()
    if sys.argv[1] == '--test':
        run_test()
    elif sys.argv[1] == '--pca':
        run_sparse_pca()
    elif sys.argv[1] == '--gmm':
        run_gmm_existing()
    else:
        run_all_sparse()
        
