import glob
import datasets

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset

class SequenceClassificationDataset(Dataset):    
    def __init__(self, raw_dataset, strlabel2int, tokenizer):
        self.inputs = raw_dataset['text']
        self.labels = raw_dataset['label']
        self.tokenizer = tokenizer
        self.strlabel2int = strlabel2int

    def __getitem__(self, idx):
        item = self.tokenizer(self.inputs[idx], padding=True, truncation=True)
        item['labels'] = torch.tensor(self.strlabel2int[self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def load_dataset(dataset, task, lang, base_path='./data'):
    data_files = {}
    for path in glob.glob(f'{base_path}/{dataset}-{task}-{lang}-*.csv'):
        split = path.split('-')[-1][:-4]
        data_files[split] = path
    return datasets.load_dataset('csv', data_files=data_files)

def load_sequence_classification_dataset(raw_datasets, strlabel2int, tokenizer, num_sample=-1, random_seed=0):
    # Load dataset
    train_dataset = None
    
    # TODO: Handle few-shot setting
    # if num_sample != -1:
    #     # Sample n-way k-shot
    #     train_df = pd.read_csv(f'{dataset_path}/train.csv').groupby('label').sample(num_sample, random_state=random_seed)
    #     train_dset = HFDataset.from_pandas(train_df.set_index('id'))    
    #     train_data = SequenceClassificationDataset(train_dset, strlabel2int, tokenizer)
    # else:
       
    train_data = SequenceClassificationDataset(raw_datasets['train'], strlabel2int, tokenizer)    
    valid_data = SequenceClassificationDataset(raw_datasets['valid'], strlabel2int, tokenizer)
    test_data = SequenceClassificationDataset(raw_datasets['test'], strlabel2int, tokenizer)
    return train_data, valid_data, test_data

if __name__ == '__main__':
    DATASET_NAMES = ['nusa_kalimat', 'nusa_alinea']
    DATASET_TO_TASKS = {
        'nusa_kalimat': ['emot', 'mt', 'senti'],
        'nusa_alinea': ['emot', 'paragraph', 'topic']
    }
    
    DATASET_TO_LANGS = {
        'nusa_kalimat': ['abs', 'btk', 'bew', 'bhp', 'jav', 'mad', 'mak', 'min', 'mui', 'rej', 'sun'],
        'nusa_alinea': ['btk', 'bew', 'bug', 'jav', 'mad', 'mak', 'min', 'mui', 'rej', 'sun']
    }
    
    dsets = {}
    for dataset_name in DATASET_NAMES:
        tasks = DATASET_TO_TASKS[dataset_name]
        langs = DATASET_TO_LANGS[dataset_name]
        for task in tasks:
            for lang in langs:
                dsets[f'{dataset_name}_{task}_{lang}'] = load_dataset(dataset_name, task, lang)
                
    print('LENGTH:', len(dsets))
    print('KEYS:', list(dsets.keys()))
    print()
    print(dsets)
