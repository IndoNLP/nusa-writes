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
        data_path = f'../data/nusa_kalimat-mt-{lang}-test.csv'
        df = pd.read_csv(data_path)
        df['ind_text'].to_csv(f'stif-indonesia/data/{lang}/{split_set}.ind', header=None, index=False)
        df['tgt_text'].to_csv(f'stif-indonesia/data/{lang}/{split_set}.tgt', header=None, index=False)