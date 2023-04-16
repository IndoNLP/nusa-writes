import numpy as np
import pandas as pd

import glob
import datasets
import torch

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset

class SequenceClassificationDataset(Dataset):    
    def __init__(self, raw_dataset, strlabel2int, tokenizer, text_col, label_col):
        self.inputs = raw_dataset[text_col]
        self.labels = raw_dataset[label_col]
        self.tokenizer = tokenizer
        self.strlabel2int = strlabel2int

    def __getitem__(self, idx):
        item = self.tokenizer(self.inputs[idx], padding=True, truncation=True)
        item['labels'] = torch.tensor(self.strlabel2int[self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def load_dataset(dataset, task, lang, num_sample:int=-1, base_path='./data'):
    data_files = {}
    for path in glob.glob(f'{base_path}/{dataset}-{task}-{lang}-*.csv'):
        split = path.split('-')[-1][:-4]
        data_files[split] = path
        #add path arguments to enable sampled data collection
    output_dataset = datasets.load_dataset('csv', data_files=data_files)
    if num_sample == -1:
      return output_dataset
    else:
      return output_dataset.filter(lambda _, idx: idx < num_sample, with_indices=True)

def load_sequence_classification_dataset(raw_datasets, strlabel2int, tokenizer, text_col, label_col, num_sample=-1, random_seed=0):
    # Load dataset
    train_dataset = None
    
    # TODO: Handle few-shot setting
    # if num_sample != -1:
    #     # Sample n-way k-shot
    #     train_df = pd.read_csv(f'{dataset_path}/train.csv').groupby('label').sample(num_sample, random_state=random_seed)
    #     train_dset = HFDataset.from_pandas(train_df.set_index('id'))    
    #     train_data = SequenceClassificationDataset(train_dset, strlabel2int, tokenizer)
    # else:
       
    train_data = SequenceClassificationDataset(raw_datasets['train'], strlabel2int, tokenizer, text_col, label_col)    
    valid_data = SequenceClassificationDataset(raw_datasets['valid'], strlabel2int, tokenizer, text_col, label_col)
    test_data = SequenceClassificationDataset(raw_datasets['test'], strlabel2int, tokenizer, text_col, label_col)
    return train_data, valid_data, test_data

def load_sequence_classification_lolo_dataset(dataset, task, test_lang, strlabel2int, tokenizer, text_col, label_col, num_sample=-1, random_seed=0, base_path="./data"):
    # Load train & validation dataset
    train_datasets = []
    valid_datasets = []
    for train_path in glob.glob(f'{base_path}/{dataset}-{task}-*-train.csv'):
        if '-all-' in train_path or f'-{test_lang}-' in train_path:
            continue
        
        valid_path = train_path.replace('train.csv','valid.csv')
        train_df = pd.read_csv(train_path)
        if num_sample != -1:
            train_df = train_df.groupby('label').sample(num_sample, random_state=random_seed)
        valid_df = pd.read_csv(valid_path)

        train_datasets.append(HFDataset.from_pandas(train_df.set_index('id')))
        valid_datasets.append(HFDataset.from_pandas(valid_df.set_index('id')))
    train_data = SequenceClassificationDataset(concatenate_datasets(train_datasets), strlabel2int, tokenizer, text_col, label_col)
    valid_data = SequenceClassificationDataset(concatenate_datasets(valid_datasets), strlabel2int, tokenizer, text_col, label_col)
        
    # Load test dataset
    test_df = pd.read_csv(f'{base_path}/{dataset}-{task}-{test_lang}-test.csv')
    test_data = SequenceClassificationDataset(HFDataset.from_pandas(test_df.set_index('id')), strlabel2int, tokenizer, text_col, label_col)    

    return train_data, valid_data, test_data

##
# Machine Translation
##
class MachineTranslationDataset(Dataset):
    # Data Schema (CSV Format)
    # id,	ind_text,	tgt_text
    # emot-train-2361,	"gimana ya caranya ngerubah instagram story viewers? kok toxic people bisa ada di urutan pertama -_- padahal buka ignya aja, saya gapernah -_-",	"piye yo corone ngobah penonton Instagram Story? kok wong-wong toksik iso ono neng urutan pertama -_- senadyan buka Instagrame wae, aku ora pernah -_-"

    def __init__(self, raw_dataset, tokenizer, swap_source_target:bool=False, *args, **kwargs):
        #slice the cols by removing unused ftrs from src_data and tgt_data
        self.src_data, self.tgt_data = raw_dataset.remove_columns(["tgt_text"]), raw_dataset.remove_columns(["ind_text"])
        self.tokenizer = tokenizer
        self.swap_source_target = swap_source_target
    
    def __getitem__(self, index):
        src_data = self.src_data[index]
        tgt_data = self.tgt_data[index]
        src_id, text = src_data['id'], src_data['ind_text']
        tgt_id, label = tgt_data['id'], tgt_data['tgt_text']
        assert src_id == tgt_id

        input_subwords = self.tokenizer.encode(text.lower(), add_special_tokens=False)
        label_subwords = self.tokenizer.encode(label.lower(), add_special_tokens=False)
        
        if self.swap_source_target:
            return src_id, label_subwords, input_subwords
        else:
            return src_id, input_subwords, label_subwords
    
    def __len__(self):
        return len(self.src_data)

###
# Generation Data Loader
###
class GenerationDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, src_lid_token_id=1, tgt_lid_token_id=2, label_pad_token_id=-100, model_type='indo-bart', tokenizer=None, *args, **kwargs):
        super(GenerationDataLoader, self).__init__(*args, **kwargs)
    
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.src_lid_token_id = src_lid_token_id
        self.tgt_lid_token_id = tgt_lid_token_id
        self.label_pad_token_id = label_pad_token_id
               
        if model_type == 'transformer':
            self.collate_fn = self._bart_collate_fn
        elif model_type == 'indo-bart':
            self.collate_fn = self._bart_collate_fn
        elif model_type == 'indo-t5':
            self.T5_TOKEN_ID_TO_LANG_MAP = {
                tokenizer.indonesian_token_id: 'indonesian',
                tokenizer.english_token_id: 'english',
                tokenizer.sundanese_token_id: 'sundanese',
                tokenizer.javanese_token_id: 'javanese'
            }
            
            if self.tokenizer is not None:
                src_lang, tgt_lang = self.T5_TOKEN_ID_TO_LANG_MAP[src_lid_token_id], self.T5_TOKEN_ID_TO_LANG_MAP[tgt_lid_token_id]
                self.t5_prefix =np.array(self.tokenizer.encode(f'translate {src_lang} to {tgt_lang}: ', add_special_tokens=False))
            
            self.collate_fn = self._t5_collate_fn
        elif model_type == 'indo-gpt2':
            self.collate_fn = self._gpt2_collate_fn
        elif model_type == 'baseline-mbart':
            self.collate_fn = self._baseline_mbart_collate_fn
        elif model_type == 'baseline-mt5':
            self.collate_fn = self._baseline_mt5_collate_fn
        else:
            raise ValueError(f'Unknown model_type `{model_type}`')
            
    def _bart_collate_fn(self, batch):
        ####
        # We make a slight format error during the pre-training, our pretrain decoder format is '<langid><bos><sent><eos>',
        #   but to ensure we have same number of prediction subword limit with mBART model (especially for summarization) 
        #   and also compatibility with other library, we then decide the resulting format will be same as mBART standard:
        # encoder input
        # <sent><eos><langid>
        # decoder input - 
        # <langid><sent><eos>
        # decoder output
        # <sent><eos><langid>
        ###
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len, max(map(lambda x: len(x[1]), batch)) + 2) # +2 for eos, and langid
        max_dec_len = min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 2) # +2 for eos, and langid
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
            input_seq, label_seq = input_seq[:max_enc_len-2], label_seq[:max_dec_len - 2]
            
            # Assign content
            enc_batch[i,0:len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + 2] = 1
            dec_mask_batch[i,:len(label_seq) + 2] = 1
            
            # Assign special token to encoder input
            enc_batch[i,len(input_seq)] = self.eos_token_id
            enc_batch[i,1+len(input_seq)] = self.src_lid_token_id
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.tgt_lid_token_id
            dec_batch[i,1+len(label_seq)] = self.eos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            label_batch[i,1+len(label_seq)] = self.tgt_lid_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, None, label_batch

    def _t5_collate_fn(self, batch):
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len, max(map(lambda x: len(x[1]), batch))  + len(self.t5_prefix))
        max_dec_len = min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 1)
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
            input_seq, label_seq = input_seq[:max_enc_len - len(self.t5_prefix)], label_seq[:max_dec_len - 1]
            
            # Assign content
            enc_batch[i,len(self.t5_prefix):len(self.t5_prefix) + len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + len(self.t5_prefix)] = 1
            dec_mask_batch[i,:len(label_seq) + 1] = 1
            
            # Assign special token to encoder input
            enc_batch[i,:len(self.t5_prefix)] = self.t5_prefix
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.bos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, None, label_batch
#         return id_batch, enc_batch, dec_batch, enc_mask_batch, dec_mask_batch, label_batch

    def _gpt2_collate_fn(self, batch): 
        ####
        # GPT2 decoder only format:
        # Training  : <src_sent><bos><tgt_sent><eos>
        # Inference : <src_sent><bos>
        #
        # Training sequence & mask are stored in dec_batch and dec_mask_batch respectively
        # Inference sequence & mask are stored in enc_batch and enc_mask_batch respectively
        ###
        batch_size = len(batch)
        max_enc_len = np.int32(min(self.max_seq_len, max(map(lambda x: len(x[1]), batch)) + 1))
        max_dec_len = np.int32(min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 1))
        max_len = max_enc_len + max_dec_len
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_len), self.pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_len), 0, dtype=np.float32)
        label_batch = np.full((batch_size, max_len), self.label_pad_token_id, dtype=np.int64) 
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
#             if max_len == self.max_seq_len:
#                 input_seq = input_seq[:max_len-len(label_seq)]
            
#             # Assign content & special token to encoder batch (for inference)
#             enc_batch[i,0] = self.bos_token_id
#             enc_batch[i,1:1 + len(input_seq)] = input_seq
#             enc_batch[i,1 + len(input_seq)] = self.eos_token_id
#             enc_batch[i,1 + len(input_seq) + 1] = self.bos_token_id
                                                
#             # Assign content & special token to decoder batch (for training)
#             # dec_batch[i,0] = self.src_lid_token_id
#             dec_batch[i,0] = self.bos_token_id
#             dec_batch[i,1:1 + len(input_seq)] = input_seq
#             dec_batch[i,1 + len(input_seq)] = self.eos_token_id
#             # dec_batch[i,1 + len(input_seq) + 1] = self.tgt_lid_token_id
#             dec_batch[i,1 + len(input_seq) + 1] = self.bos_token_id
#             dec_batch[i,1 + len(input_seq) + 2: 1 + len(input_seq) + 2 + len(label_seq)] = label_seq
                                                
#             # Assign Mask for encoder & decoder batch
#             enc_mask_batch[i,:1 + len(input_seq) + 2] = 1
#             dec_mask_batch[i,:1 + len(input_seq) + 2 + len(label_seq)] = 1
            
#             # Assign content & special token to label batch, ignore the input prefix until <tgt_lang_id>
#             label_batch[i,1 + len(input_seq) + 1:1 + len(input_seq) + 1 + len(label_seq)] = label_seq
#             label_batch[i,1 + len(input_seq) + 1 + len(label_seq)] = self.eos_token_id

            input_seq = input_seq[:self.max_seq_len - 1]
            label_seq = label_seq[:self.max_seq_len - 1]

            # Assign content & special token to encoder batch (for inference)
            enc_batch[i,:len(input_seq)] = input_seq
            enc_batch[i,max_enc_len - 1] = self.bos_token_id
                                                
            # Assign content & special token to decoder batch (for training)
            # dec_batch[i,0] = self.src_lid_token_id
            dec_batch[i,:len(input_seq)] = input_seq
            dec_batch[i,max_enc_len-1] = self.bos_token_id
            # dec_batch[i,1 + len(input_seq) + 1] = self.tgt_lid_token_id
            dec_batch[i,max_enc_len:max_enc_len+len(label_seq)] = label_seq
                                                
            # Assign Mask for encoder batch
            enc_mask_batch[i,:len(input_seq)] = 1
            enc_mask_batch[i,max_enc_len - 1] = 1
            
            # Assign Mask for decoder batch
            dec_mask_batch[i,:len(input_seq)] = 1
            dec_mask_batch[i,max_enc_len - 1] = 1
            dec_mask_batch[i,max_enc_len:max_enc_len+len(label_seq)] = 1
            
            # Assign content & special token to label batch, no need to shift left as it will be done inside the GPT2LMHeadModel
            label_batch[i,max_enc_len:max_enc_len+len(label_seq)] = label_seq
            label_batch[i,max_enc_len+len(label_seq)] = self.eos_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, dec_mask_batch, label_batch

    def _baseline_mbart_collate_fn(self, batch):
        ####
        # We follow mBART pre-training format, there is a discussions for the mBART tokenizer (https://github.com/huggingface/transformers/issues/7416)
        #   which mentioned the format of the labels should be: <langid><sent><eos><langid>
        #   and the mBART model will add the <langid> as a prefix to create the decoder_input_ids during the forward function.
        # 
        # In order to make it consistent and easier to understand with the other models, we keep our dataloader similar to our IndoNLG models
        #   with the following output format:
        # encoder input
        # <sent><eos><langid>
        # decoder input
        # <langid><sent><eos>
        # decoder output
        # <sent><eos><langid>
        ###
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len, max(map(lambda x: len(x[1]), batch)) + 2) # + 2 for eos and langid
        max_dec_len = min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 2) # + 2 for eos and langid
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
            input_seq, label_seq = input_seq[:max_enc_len-2], label_seq[:max_dec_len - 2]
            
            # Assign content
            enc_batch[i,0:len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,0:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + 2] = 1
            dec_mask_batch[i,:len(label_seq) + 2] = 1
            
            # Assign special token to encoder input
            enc_batch[i,len(input_seq)] = self.eos_token_id
            enc_batch[i,1+len(input_seq)] = self.src_lid_token_id
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.tgt_lid_token_id
            dec_batch[i,1+len(label_seq)] = self.eos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            label_batch[i,1+len(label_seq)] = self.tgt_lid_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, None, label_batch
#         return id_batch, enc_batch, dec_batch, enc_mask_batch, dec_mask_batch, label_batch
        
    def _baseline_mt5_collate_fn(self, batch):
        ####
        # As mT5 is only trained on MLM without additional language identifier, we can actually fine tune without prefix
        # In this case we make the input output format as follow
        # encoder input
        # <sent>
        # decoder input
        # <bos><sent>
        # decoder output
        # <sent><eos>
        ###
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len, max(map(lambda x: len(x[1]), batch))) # No additional token is needed
        max_dec_len = min(self.max_seq_len, max(map(lambda x: len(x[2]), batch)) + 1) # + 1 for bos / eos
        
        id_batch = []
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, (id, input_seq, label_seq) in enumerate(batch):
            input_seq, label_seq = input_seq[:max_enc_len], label_seq[:max_dec_len - 1]
            
            # Assign content
            enc_batch[i,0:len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,0:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq)] = 1
            dec_mask_batch[i,:len(label_seq) + 1] = 1
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.bos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            
            id_batch.append(id)
        
        return id_batch, enc_batch, dec_batch, enc_mask_batch, None, label_batch


if __name__ == '__main__':
    DATASET_NAMES = ['nusa_kalimat', 'nusa_alinea']
    DATASET_TO_TASKS = {
        'nusa_kalimat': ['emot', 'mt', 'senti'],
        'nusa_alinea': ['emot', 'paragraph', 'topic', 'author']
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
                try:
                    dsets[f'{dataset_name}_{task}_{lang}'] = load_dataset(dataset_name, task, lang)
                except:
                    print(f'No data for task `{task}` on lang `{lang}` in the {dataset_name} dataset')
                    pass
                
    print('LENGTH:', len(dsets))
    print('KEYS:', list(dsets.keys()))
    print()
    print(dsets)
