CONFIG_TO_PROMPT = {
    'sentimix_spaeng': [
        '[INPUT] => Sentiment: [LABELS_CHOICE]',
        'Text: [INPUT] => Sentiment: [LABELS_CHOICE]',
        '[INPUT]\nWhat would be the sentiment of the text above? [LABELS_CHOICE]',
        'What is the sentiment of this text?\nText: [INPUT]\nAnswer: [LABELS_CHOICE]',
        'Text: [INPUT]\nPlease classify the sentiment of above text. Sentiment: [LABELS_CHOICE]',
    ],
    'tamil_mixsentiment': [
        '[INPUT] => Sentiment: [LABELS_CHOICE]',
        'Text: [INPUT] => Sentiment: [LABELS_CHOICE]',
        '[INPUT]\nWhat would be the sentiment of the text above? [LABELS_CHOICE]',
        'What is the sentiment of this text?\nText: [INPUT]\nAnswer: [LABELS_CHOICE]',
        'Text: [INPUT]\nPlease classify the sentiment of above text. Sentiment: [LABELS_CHOICE]',
    ], 
    'malayalam_mixsentiment': [
        '[INPUT] => Sentiment: [LABELS_CHOICE]',
        'Text: [INPUT] => Sentiment: [LABELS_CHOICE]',
        '[INPUT]\nWhat would be the sentiment of the text above? [LABELS_CHOICE]',
        'What is the sentiment of this text?\nText: [INPUT]\nAnswer: [LABELS_CHOICE]',
        'Text: [INPUT]\nPlease classify the sentiment of above text. Sentiment: [LABELS_CHOICE]',
    ],
    'mt_enghinglish':[
        'Translate the following text from [SOURCE] to [TARGET].\nText: [INPUT]\nTranslation:',
        '[INPUT]\nTranslate the text above from [SOURCE] to [TARGET].',
        'Text in [SOURCE]: [INPUT]\nHow would you translate that in [TARGET]?',
        'Translate the following [SOURCE] text from to [TARGET].\nText: [INPUT]\nTranslation:',
        'Text in [SOURCE]: [INPUT]\nText in [TARGET]:',
    ],

    "lid":[
        '[INPUT] =>',
        "[INPUT] \n For each word in the text above, assign a language tag (lang1, lang2, or none) using the format [ word | tag ].",
        "[INPUT] \n Annotate each word in the text above as [ word | tag ], using tags: lang1, lang2, or none.",
        "[INPUT] \n Assign [ word | tag ] to each word in the text above with tags lang1, lang2, or none.",
        "[INPUT] \n Indicate the language of each word by tagging it as [ word | tag ], choosing from lang1, lang2, or none."
    ]
}
def get_prompt(config_names):
    prompt_templates = {}
    for config, prompts in CONFIG_TO_PROMPT.items():
        if config in config_names:
            prompt_templates[config] = prompts
    return prompt_templates
