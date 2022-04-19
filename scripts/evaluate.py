import pickle
import pandas as pd
import statistics
from scipy.stats import iqr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open('test_frame.pkl', 'rb')as test_in, open('generated_results.pkl', 'rb') as results_in:
    test_frame = pickle.load(test_in)
    results_list = pickle.load(results_in)

    tested_frame = pd.DataFrame(test_frame[:len(results_list)])
    tested_frame['generated'] = results_list


    def evaluate_bleu():
        
        from nltk.translate.bleu_score import sentence_bleu

        bleu_scores = []

        for entry_index in range(len(tested_frame)):
        
            true_label = tested_frame['simple'][entry_index].split()
            generated_label = tested_frame['generated'][entry_index].split()

            pair_bleu = sentence_bleu([true_label], generated_label)
            bleu_scores.append(pair_bleu)

    
        # tested_frame['bleu'] = bleu_scores
        
        mean_bleu_score = statistics.mean(bleu_scores)
        median_bleu_score = statistics.median(bleu_scores)
        max_bleu_score = max(bleu_scores)
        min_bleu_score = min(i for i in bleu_scores if i > 0)
        bleu_iqr = iqr(bleu_scores)    

        return(mean_bleu_score, median_bleu_score, max_bleu_score, min_bleu_score, bleu_iqr, bleu_scores)


    def evaluate_sari():#frame: pd.DataFrame) -> pd.DataFrame([['normal (sources)', 'simple (references)', 'generated (predictions)']]):
        
        from datasets import load_metric

        sari = load_metric('sari')
        sari_scores = []

        for entry_index in range(len(tested_frame)):

            source = tested_frame['normal'][entry_index]
            reference = tested_frame['simple'][entry_index]
            prediction = tested_frame['generated'][entry_index]

            result = sari.compute(sources=[source], predictions=[prediction], references=[[reference]])
            sari_scores.append(result['sari'])

            # DEBUG
            # print(f"{entry_index}/{len(tested_frame)}: SARI {result}")


        # tested_frame['sari'] = sari_scores
        
        mean_sari_score = statistics.mean(sari_scores)
        median_sari_score = statistics.median(sari_scores)
        max_sari_score = max(sari_scores)
        min_sari_score = min(i for i in sari_scores if i > 0)
        sari_iqr = iqr(sari_scores)    

        return mean_sari_score, median_sari_score, max_sari_score, min_sari_score, sari_iqr, sari_scores



    sari_values = evaluate_sari()


    blue_values = evaluate_bleu()

    print(f'\
        mean: {blue_values[0]}\n\
        median: {blue_values[1]}\n\
        max: {blue_values[2]}\n\
        min: {blue_values[3]}\n\
        iqr: {blue_values[4]}')

    print(f'\
            mean: {sari_values[0]}\n\
            median: {sari_values[1]}\n\
            max: {sari_values[2]}\n\
            min: {sari_values[3]}\n\
            iqr: {sari_values[4]}')


    
    # sari_plot_data = sari_values[5]
    # plt.hist(sari_plot_data)
    # plt.yscale('log')
    # plt.savefig('sari_histogram.png', bbox_inches='tight')

    # bleu_plot_data = blue_values[5]
    # plt.hist(bleu_plot_data)
    # plt.savefig('bleu_histogram', bbox_inches='tight')

    # sns.set_style('whitegrid')
    # bleu_plot = sns.kdeplot(bleu_plot_data, bw=0.5)
    # bleu_fig = bleu_plot.get_figure()
    # bleu_fig.savefig("bleu_density.png")

    # sari_plot = sns.kdeplot(sari_plot_data, bw=0.5)
    # sari_fig = sari_plot.get_figure()
    # sari_fig.savefig("sari_density.png")

    
    # fig = swarm_plot.get_figure()
    # fig.savefig("out.png") 
    # plt.boxplot(plot_data)

    # plt.yscale('log')
    # plt.savefig('sari_boxplot_logspace.png', bbox_inches='tight')
    

    