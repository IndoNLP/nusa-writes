TASK_TO_PROMPT = {
    'emot': [
        '[INPUT] => Emotion: [LABELS_CHOICE]',
        'Text: [INPUT] => Emotion: [LABELS_CHOICE]',
        '[INPUT]\nWhat would be the emotion of the text above? [LABELS_CHOICE]',
        'What is the emotion of this text?\nText: [INPUT]\nAnswer: [LABELS_CHOICE]',
        'Text: [INPUT]\nPlease classify the emotion of above text. Emotion: [LABELS_CHOICE]',
    ],
    'paragraph': [
        '[INPUT] => Rhetorical mode: [LABELS_CHOICE]',
        'Text: [INPUT] => Rhetorical mode: [LABELS_CHOICE]',
        '[INPUT]\nWhat would be the rhetorical mode of the text above? [LABELS_CHOICE]',
        'What is the rhetorical mode of this text?\nText: [INPUT]\nAnswer: [LABELS_CHOICE]',
        'Text: [INPUT]\nPlease classify the rhetorical mode of above text. Rhetorical mode: [LABELS_CHOICE]',
    ],
    'topic':  [
        '[INPUT] => Topic: [LABELS_CHOICE]',
        'Text: [INPUT] => Topic: [LABELS_CHOICE]',
        '[INPUT]\nWhat would be the topic of the text above? [LABELS_CHOICE]',
        'What is the topic of this text?\nText: [INPUT]\nAnswer: [LABELS_CHOICE]',
        'Text: [INPUT]\nPlease classify the topic of above text. Topic: [LABELS_CHOICE]',
    ],
    'senti': [
        '[INPUT] => Sentiment: [LABELS_CHOICE]',
        'Text: [INPUT] => Sentiment: [LABELS_CHOICE]',
        '[INPUT]\nWhat would be the sentiment of the text above? [LABELS_CHOICE]',
        'What is the sentiment of this text?\nText: [INPUT]\nAnswer: [LABELS_CHOICE]',
        'Text: [INPUT]\nPlease classify the sentiment of above text. Sentiment: [LABELS_CHOICE]',
    ],
    'mt': [
        'Translate the following text from [SOURCE] to [TARGET].\nText: [INPUT]\nTranslation:',
        '[INPUT]\nTranslate the text above from [SOURCE] to [TARGET].',
        'Text in [SOURCE]: [INPUT]\nHow would you translate that in [TARGET]?',
        'Translate the following [SOURCE] text from to [TARGET].\nText: [INPUT]\nTranslation:',
        'Text in [SOURCE]: [INPUT]\nText in [TARGET]:',
    ]
}
def get_prompt():
    prompt_templates = {}
    for config, prompts in TASK_TO_PROMPT.items():
        prompt_templates[config] = prompts
    return prompt_templates
