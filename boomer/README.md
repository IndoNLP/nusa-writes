## Usage
Make sure you workring directory already in `nusa-menulis/boomer`

To evaluate the datasets, you can run the following command:

```bash
python main.py --dataset_name DATASET_NAME --tasks TASKS --langs LANGS
```
Replace DATASET_NAME, TASKS and LANGS with the names of the dataset, tasks and languages you want to evaluate. You can use the --dataset_name, --tasks and --lang arguments to specify multiple datasets, tasks and languages to evaluate.

If you want to evaluate all datasets, tasks and languages, you can run:

```bash
python main.py
```
The evaluation results will be saved to a CSV file named test_results.csv in the boomer directory.

## Example
To evaluate the nusa_alinea dataset for the Authorship Identification task and the Betawi language, you can run:

```bash
python main.py --dataset_name nusa_alinea --tasks author --langs bew
```
To evaluate the nusa_alinea and nusa_menulis datasets for the Emotion Classifcation task and the Betawi and Batak languages, you can run:

```bash
python main.py --dataset_name nusa_alinea,nusa_menulis --tasks emot --langs bew,btk
```