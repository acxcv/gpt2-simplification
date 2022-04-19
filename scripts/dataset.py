import pandas as pd
# from pprint import pprint
import pickle

with pd.option_context('display.max_rows', None):

    dataset = pd.read_csv('merged_data.tsv', sep='\t')

    text_dict = {}


    for row in dataset.iterrows():
        data = row[1]
        entry = data[0]
        a_key = data[1] 
        normal  = data[2]
        simple = data[3]

        if entry in text_dict.keys():
            if a_key not in text_dict[entry].keys():
                text_dict[entry].update({a_key: {'normal': [], 'simple': []}})
            
            normal_list = text_dict[entry][a_key]['normal']
            simple_list = text_dict[entry][a_key]['simple']

            if normal not in normal_list:
                text_dict[entry][a_key]['normal'].append(normal)
            if simple not in simple_list:
                text_dict[entry][a_key]['simple'].append(simple)
        else:
            text_dict[entry] = {a_key: {'normal': [normal], 'simple': [simple]}}


    with open('data_dict_lists.pkl', 'wb+') as f:
            pickle.dump(text_dict, f)
    