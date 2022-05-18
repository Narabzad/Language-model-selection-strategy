# Language-model-selection-strategy

This repository include the code and data regarding the paper "A Pre-trained Language Model Selection Strategy for Biomedical Question Answering". In this paper, we propose a classifier to select appropriate general-purpose on per input basis to address domain-specific downstream NLP tasks.

To replicate our results, clone this repository and follow the steps below. 

1.```fine-tune.py``` script would fine tune the following models on BioASQ7b dataset on question answering task.
- [BERT](https://huggingface.co/bert-base-uncased)
- [Roberta](https://huggingface.co/roberta-base)
- [Distilbert](https://huggingface.co/distilbert-base-uncased)
- [DistilRoberta](https://huggingface.co/distilroberta-base)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)

You can set the training parameter such as [```num_epoch```](https://github.com/Narabzad/Language-model-selection-strategy/blob/85a02ed4294412db32e1315a9017ae264a6103b2/fine-tune.py#L39) in ```fine-tune.py```
The fine-tuned models and their results would be saved in [```models/fine-tuned```](https://github.com/Narabzad/Language-model-selection-strategy/tree/main/models/fine-tuned) and [```results/fine-tuned```](https://github.com/Narabzad/Language-model-selection-strategy/tree/main/results/fine-tuned) respectively.

After you have the standalone model's results, you can train a classifier based on these models evalaution on BioASQ8b train set using ```train_classifier.py```.
```train_classifier.py``` would train classifier on every pair of the mentioned language models and save the classifiers in [```models/classifier```](https://github.com/Narabzad/Language-model-selection-strategy/tree/main/models/classifier).

Due to size of the models, we uploaded our best-performened classifiers [here](https://drive.google.com/drive/folders/19qkRvFTkn1MbJWNjymLlK9rGLYcRjjgm?usp=sharing).

In the last step, you may get the final prediction of our proposed appraoch using ```test_classifier.py```. The results of prediction would be stored in [```results/ours```](https://github.com/Narabzad/Language-model-selection-strategy/tree/main/results/ours) in the following format :

```question_id \t oracle_lm \t predicted_lm ```
