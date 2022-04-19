import pandas as pd
import pickle

with open('data_dict_lists.pkl', 'rb') as infile:
    dataset_dict = pickle.load(infile)


data_df = pd.DataFrame(columns=['entry','normal','simple'])

for entry in dataset_dict.items():
    
    entry_name = entry[0]
    entry_dict = entry[1]

    for subdict in entry_dict.items():

        a_key = subdict[0]
        entry_text = subdict[1]


        # print(subdict)
        new_row = {
            'entry': f'{entry_name}-{a_key}', 
            'normal': ' '.join([str(sentence) for sentence in entry_text['normal']]), 
            'simple': ' '.join([str(sentence) for sentence in entry_text['simple']])}

        data_df = data_df.append(new_row, ignore_index=True)

    # debug
    # print(data_df)


with open('dataset_final.pkl', 'wb') as outfile:
    pickle.dump(data_df, outfile)