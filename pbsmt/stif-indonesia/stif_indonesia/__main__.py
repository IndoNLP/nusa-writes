import click
from .main_moses import MosesSMTModel
from .logging import CustomLogger


sup_experiments = [
    'experiment-config/00001_default_supervised_config.json'
]


experiment_semi_supervised = [
    'experiment-config/00002_default_semi_supervised_config.json',
]

nusa_menulis_train_exps = [
    'abs','btk','bew','bhp','jav','mad','mak','min','mui','rej','sun'
]

nusa_menulis_eval_exps = [
    'abs','btk','bew','bhp','jav','mad','mak','min','mui','rej','sun'
]

def do_experiment(exp):
    moses_model = MosesSMTModel(exp, use_wandb=False)
    moses_model.run_experiments()

def do_semi_supervised_experiment(exp):
    moses_model = MosesSMTModel(exp, use_wandb=False)
    moses_model.run_semi_supervised()

def do_nusa_menulis_train(exp):
    exp_config = f'experiment-config/{exp}.json'
    moses_model = MosesSMTModel(exp_config, use_wandb=False)
    moses_model.run_nusa_menulis_train(exp)

def do_nusa_menulis_eval(exp):
    exp_config = f'experiment-config/{exp}.json'
    moses_model = MosesSMTModel(exp_config, use_wandb=False)
    moses_model.run_nusa_menulis_eval(exp)

@click.command()
@click.option('--exp-scenario', help='possible "supervised" or "semi-supervised"')
def main(exp_scenario: str):
    """
    To customize your needs, add your experiment config and put it to the 'sup_experiments' or 'experiment_semi_supervised'
    """
    CustomLogger().create_logger('moses-rerun', log_file='log.log', alay=True)
    if exp_scenario == 'supervised':
        for exp in sup_experiments:
            do_experiment(exp)
    if exp_scenario == 'semi-supervised':
        for exp in experiment_semi_supervised:
            do_semi_supervised_experiment(exp)

    if exp_scenario == 'nusa-menulis-train':
        for exp in nusa_menulis_train_exps:
            do_nusa_menulis_train(exp)

    if exp_scenario == 'nusa-menulis-eval':
        with open('results.csv','w') as f:
            f.write('lang, bleu\n') 

        for exp in nusa_menulis_eval_exps:
            do_nusa_menulis_eval(exp)

if __name__ == '__main__':
    main()
