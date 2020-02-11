import os
import pandas as pd
from semcor_bert_pipeline import *
from clustering import *
from fpdf import FPDF


def run_clustering(word, pos, model):
    print(word, pos)
    word_results = run_pipeline(word, pos, model, min_senses = 10)
    print("Generating Plots")
    tsne_results = plot_embeddings(word_results['embeddings'], word_results['sense_indices'], word_results['sense_names'], word_results['lemma'], savefile = True)
    color_dict, label_dict = create_dendrogram_colors(word_results['sense_names'])
    plot_dendrogram(word_results, color_dict, label_dict, savefile = True)
    print("Running GMM Simulations")
    raw_json = plot_gmm_rand_indices(word_results, range(2, 30), savefile = True)


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


if __name__ == '__main__':
    model = initialize_model()
    sparse_senses = pd.read_csv('data/semcor_sparsity.csv')
    for i in range(2, len(sparse_senses.index)):
        row = sparse_senses.iloc[i]
        word, pos = row['word'], row['pos']
        dir_name = os.path.join("data", 'clustering_results', word + '_' + pos)
        os.system('mkdir ' + dir_name)
        run_clustering(word, pos, model)
        convert_imgs_to_pdf(word, pos)
