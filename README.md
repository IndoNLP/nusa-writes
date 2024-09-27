# NusaWrites

### Overview
**NusaWrites** is an in-depth analysis of corpora collection strategy and a comprehensive language modeling benchmark for underrepresented and extremely low-resource Indonesian local languages.

**NusaWrites** benchmark consists of two datasets over 6 different tasks: 
1. NusaTranslation 
	- NusaTranslation is a human translated local languages dataset consists of 72,444 textual data from multiple languages
	- NusaTranslation covers 11 local languages, including Ambon (abs), Batak (btk), Betawi (bew), Bima (bhp), Javanese (jav), Madurese (mad), Makassarese (mak), Minangkabau (min), Palembang / Musi (mui), Rejang (rej), and Sundanese (sun)
	- NusaTranslation support 3 downstream tasks: sentiment analysis, emotion classification, and machine translation
	
2. NusaParagraph
	- NusaParagraph is a human paragraph writing dataset consists of 57,409 paragraphs from multiple Indonesian local languages
	- NusaParagraph covers 10 local languages, including Batak (btk), Betawi (bew), Buginese (bug), Javanese (jav), Madurese (mad), Makassarese (mak), Minangkabau (min), Palembang / Musi (mui), Rejang (rej), and Sundanese (sun)
	- NusaParagraph supports 3 downstream tasks: emotion classification, topic modeling, and rhetoric mode classification

### How to Use

The complete **NusaWrites** dataset can be accessed from our [github repository](https://github.com/IndoNLP/nusa-writes). 
For a more easy-to-use and standardized access of all **NusaWrites** datasets, you can access it though the [Hugging Face `datasets` library]() or our [NusaCrowd library]()

##### Access from Hugging Face `datasets`
```
import datasets

# NusaTranslation (all Languages mixed)
nt_emot_dset = datasets.load_dataset('indonlp/nusatranslation_emot')
nt_senti_dset = datasets.load_dataset('indonlp/nusatranslation_senti')
nt_mt_dset = datasets.load_dataset('indonlp/nusatranslation_mt')

# NusaTranslation (per language)
# Supported lang_code: abs, btk, bew, bhp, jav, mad, mak, min, mui, rej, sun
nt_emot_dset = datasets.load_dataset('indonlp/nusatranslation_emot', name='nusatranslation_emot_{lang_code}_nusantara_text')
nt_senti_dset = datasets.load_dataset('indonlp/nusatranslation_senti', name='nusatranslation_senti_{lang_code}_nusantara_text')
nt_mt_dset = datasets.load_dataset('indonlp/nusatranslation_mt', name='nusatranslation_mt_{lang_code}_nusantara_text')

# NusaParagraph (all Languages mixed)
np_emot_dset = datasets.load_dataset('indonlp/nusaparagraph_emot')
np_rhetoric_dset = datasets.load_dataset('indonlp/nusaparagraph_rhetoric')
np_topic_dset = datasets.load_dataset('indonlp/nusaparagraph_topic')

# NusaParagraph (per language)
# Supported lang_code: btk, bew, bug, jav, mad, mak, min, mui, rej, sun
np_emot_dset = datasets.load_dataset('indonlp/nusaparagraph_emot', name='nusaparagraph_emot_{lang_code}_nusantara_text')
np_rhetoric_dset = datasets.load_dataset('indonlp/nusaparagraph_rhetoric', name='nusaparagraph_rhetoric_{lang_code}_nusantara_text')
np_topic_dset = datasets.load_dataset('indonlp/nusaparagraph_topic', name='nusaparagraph_topic_{lang_code}_nusantara_text')
```

##### Access from NusaCrowd

Loading per task dataset
```
import nusacrowd as nc

# NusaTranslation (all Languages mixed)
nt_emot_dset = nc.load_dataset('nusatranslation_emot')
nt_senti_dset = nc.load_dataset('nusatranslation_senti')
nt_mt_dset = nc.load_dataset('nusatranslation_mt')

# NusaTranslation (per language)
# Supported lang_code: abs, btk, bew, bhp, jav, mad, mak, min, mui, rej, sun
nt_emot_dset = nc.load_dataset('indonlp/nusatranslation_emot', name='nusatranslation_emot_{lang_code}_nusantara_text')
nt_senti_dset = nc.load_dataset('indonlp/nusatranslation_senti', name='nusatranslation_senti_{lang_code}_nusantara_text')
nt_mt_dset = nc.load_dataset('indonlp/nusatranslation_mt', name='nusatranslation_mt_{lang_code}_nusantara_text')

# NusaParagraph (all Languages mixed)
np_emot_dset = nc.load_dataset('indonlp/nusaparagraph_emot')
np_rhetoric_dset = nc.load_dataset('indonlp/nusaparagraph_rhetoric')
np_topic_dset = nc.load_dataset('indonlp/nusaparagraph_topic')

# NusaParagraph (per language)
# Supported lang_code: btk, bew, bug, jav, mad, mak, min, mui, rej, sun
np_emot_dset = nc.load_dataset('indonlp/nusaparagraph_emot', name='nusaparagraph_emot_{lang_code}_nusantara_text')
np_rhetoric_dset = nc.load_dataset('indonlp/nusaparagraph_rhetoric', name='nusaparagraph_rhetoric_{lang_code}_nusantara_text')
np_topic_dset = nc.load_dataset('indonlp/nusaparagraph_topic', name='nusaparagraph_topic_{lang_code}_nusantara_text')
```

Loading the whole benchmark
```
# NusaTranslation
nusa_translation_dsets = nc.load_benchmark('NusaTranslation')

# NusaParagraph
nusa_paragraph_dsets = nc.load_benchmark('NusaParagraph')

# NusaWrites
nusa_writes_dsets = nc.load_benchmark('NusaWrites')
```

### Experiment Code

##### Running LM Experiment

We modify the `run_clm.py` code from Hugging Face and made use of IndoGPT (https://huggingface.co/indobenchmark/indogpt) tokenizer in our LM experiment. 
The code and the run script can be found under the [lm-exp](https://github.com/IndoNLP/nusa-writes/tree/main/lm-exp) folder in the repository.
- `run_clm.py` → https://github.com/IndoNLP/nusa-writes/blob/main/lm-exp/run_clm.py
- Bash runner script (`run_lm_exp.sh`) → https://github.com/IndoNLP/nusa-writes/blob/main/lm-exp/run_lm_exp.sh

##### Running PBSMT Experiment

To run the PBSMT experiment, you can follow the run the code in the following order:
- Generate dataset → https://github.com/IndoNLP/nusa-writes/pbsmt/convert_data.py
- Generate config → https://github.com/IndoNLP/nusa-writes/pbsmt/generate_configs.py
- Training → https://github.com/IndoNLP/nusa-writes/blob/stif-indonesia/run_nusa_menulis_train.sh
- Testing → https://github.com/IndoNLP/nusa-writes/blob/stif-indonesia/run_nusa_menulis_eval.sh 


### Research Paper
Our work has been accepted in AACL 2023 and published [here](https://aclanthology.org/2023.ijcnlp-main.60/).

If you find our work helpful, please cite the following article:
```
@inproceedings{cahyawijaya-etal-2023-nusawrites,
    title = "{N}usa{W}rites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages",
    author = "Cahyawijaya, Samuel  and  Lovenia, Holy  and Koto, Fajri  and  Adhista, Dea  and  Dave, Emmanuel  and  Oktavianti, Sarah  and  Akbar, Salsabil  and  Lee, Jhonson  and  Shadieq, Nuur  and  Cenggoro, Tjeng Wawan  and  Linuwih, Hanung  and  Wilie, Bryan  and  Muridan, Galih  and  Winata, Genta  and  Moeljadi, David  and  Aji, Alham Fikri  and  Purwarianti, Ayu  and  Fung, Pascale",
    editor = "Park, Jong C.  and  Arase, Yuki  and  Hu, Baotian  and  Lu, Wei  and  Wijaya, Derry  and  Purwarianti, Ayu  and  Krisnadhi, Adila Alfa",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = nov,
    year = "2023",
    address = "Nusa Dua, Bali",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-main.60",
    pages = "921--945",
}
```
