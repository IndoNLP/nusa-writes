import pandas as pd
import os
from tqdm import tqdm

for split_set in ['train', 'valid', 'test']:
    print(split_set)
    for lang in tqdm(['abs','btk','bew','bhp','jav','mad','mak','min','mui','rej','sun']):
        path = f'stif-indonesia/data/{lang}'
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        data_path = f'../data/nusa_kalimat-mt-{lang}-{split_set}.csv'
        df = pd.read_csv(data_path)

        split_set_new = split_set
        if split_set == 'valid':
            split_set_new = 'dev'

        df['ind_text'].to_csv(f'stif-indonesia/data/{lang}/{split_set_new}.inf', header=None, index=False)
        df['tgt_text'].to_csv(f'stif-indonesia/data/{lang}/{split_set_new}.for', header=None, index=False)