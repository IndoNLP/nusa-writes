import pandas as pd
import os
from tqdm import tqdm
import json

for lang in tqdm(['abs','btk','bew','bhp','jav','mad','mak','min','mui','rej','sun']):
    config_dict = {
        "data_dir" : f"data/{lang}/",
        "output_working_dir" : f"output/{lang}/",
        "semi-supervised-count" : 0,
        "moses_args" : {
            "moses_ngram" : 3,
            "core_cpu" : 2,
            "reordering" : "msd-bidirectional-fe",
            "alignment" : "grow-diag-final-and"
        },
        "data_train" : "train",
        "data_development" : "dev",
        "data_test" : "test",
        "source_file_type" : ".inf",
        "target_file_type" : ".for",
        "wandb_project_run": "nusa-menulis",
        "wandb_notes": f"{lang}",
        "wandb_name": f"{lang}",
        "wandb_tags": [f"{lang}"]
    }
    with open(f"stif-indonesia/experiment-config/{lang}.json", "w") as outfile:
        json.dump(config_dict, outfile, indent = 4)