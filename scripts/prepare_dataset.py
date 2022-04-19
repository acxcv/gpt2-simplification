import pandas as pd
import pickle

with open('dataset_final.pkl', 'rb') as infile:
    data_frame = pickle.load(infile)


    # split each string in the df to count tokens
    data_frame['normal_count'] = data_frame['normal'].map(lambda x: len(x.split()))
    data_frame['simple_count'] = data_frame['simple'].map(lambda x: len(x.split()))

    # remove short and long sequences (mostly noise)
    data_frame = data_frame.loc[(data_frame['normal_count'] > 5) & 
                                (data_frame['normal_count'] < 300) & 
                                (data_frame['simple_count'] > 5) & 
                                (data_frame['simple_count'] < 300)]
    print(data_frame)

    with open('final_frame_300.pkl', 'wb+') as outfile:
            pickle.dump(data_frame, outfile)
