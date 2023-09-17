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
For a more easy-to-use and standardized access of all **NusaWrites** datasets, you can access it though the [HuggingFace `datasets` library]() or our [NusaCrowd library]()

##### Access from HuggingFace `datasets`
```
import datasets

# NusaTranslation
nt_emot_dset = datasets.load_dataset('indonlp/nusatranslation_emot')
nt_senti_dset = datasets.load_dataset('indonlp/nusatranslation_senti')
nt_mt_dset = datasets.load_dataset('indonlp/nusatranslation_mt')

# NusaParagraph
np_emot_dset = datasets.load_dataset('indonlp/nusaparagraph_emot')
np_rhetoric_dset = datasets.load_dataset('indonlp/nusaparagraph_rhetoric')
np_topic_dset = datasets.load_dataset('indonlp/nusaparagraph_topic')
```

##### Access from NusaCrowd

Loading per task dataset
```
import nusacrowd as nc

# NusaTranslation
nt_emot_dset = nc.load_dataset('nusatranslation_emot')
nt_senti_dset = nc.load_dataset('nusatranslation_senti')
nt_mt_dset = nc.load_dataset('nusatranslation_mt')

# NusaParagraph
np_emot_dset = nc.load_dataset('nusaparagraph_emot')
np_rhetoric_dset = nc.load_dataset('nusaparagraph_rhetoric')
np_topic_dset = nc.load_dataset('nusaparagraph_topic')
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

We modify the `run_clm.py` code from Huggingface and made use of IndoGPT (https://huggingface.co/indobenchmark/indogpt) tokenizer in our LM experiment. 
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
Our work has been accepted in AACL 2023 and is currently waiting to be published. In the meantime, you can access the preprint version of our work [here](https://openreview.net/forum?id=gftlYED4KRp). 

If you find our work useful, please cite the following article:
```
@unpublished{              
	cahyawijaya2023nusawrites,              
	title={NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages},              
	author={Samuel Cahyawijaya,Holy Lovenia,Fajri Koto,Dea Adhista,Emmanuel Dave,Sarah Oktavianti,Salsabil Maulana Akbar,Jhonson Lee,Nuur Shadieq,Tjeng Wawan Cenggoro,Hanung Linuwih,Bryan Wilie,Galih Pradipta Muridan,Alham Fikri Aji,Genta Indra Winata,David Moeljadi,Ayu Purwarianti,Pascale Fung},              
	journal={OpenReview Preprint},              
	year={2023},              
	note={anonymous preprint under review}          
}
```
