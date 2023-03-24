import glob
import datasets

def load_dataset(dataset, task, lang, base_path='./data'):
    data_files = {}
    for path in glob.glob(f'{base_path}/{dataset}-{task}-{lang}-*.csv'):
        split = path.split('-')[-1][:-4]
        data_files[split] = path
    if len(data_files) == 0:
        print('Nah loh', dataset, task, lang)
    return datasets.load_dataset('csv', data_files=data_files)

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
