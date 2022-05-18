# Language-model-selection-strategy

This repository include the code and data regarding the paper "A Pre-trained Language Model Selection Strategy for Biomedical Question Answering". In this paper, we propose a classifier to select appropriate general-purpose on per input basis to address domain-specific downstream NLP tasks.

To replicate our results, clone this repository and follow the steps below. 

1.``fine-tune.py'' script would fine tune the following models on BioASQ7b dataset on question answering task.
- [BERT](https://huggingface.co/bert-base-uncased)
- [Roberta](https://huggingface.co/roberta-base)
- [Distilbert](https://huggingface.co/distilbert-base-uncased)
- [DistilRoberta](https://huggingface.co/distilroberta-base)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)

You can set the training parameter such as ``epoch_num'' in fine-tune.py
The results of standalone model's prediction wouldd be saved in models/standalone-fine-tuned

After you have the standalone model results, you can train a classifier based on these models evalaution on BioASQ8b train set using ``train_classifier.py''
``train_classifier.py'' would train classifier on each pari of the mentioned model and save the classifier models in 1``'models/classifier''.

Due to size of the models, we uploaded our best-performened classifiers in this google drive.

In the last stepo, you may get the final prediction of our proposed appraoch using 11'test_classifier.py''. The results of prediction would be stored in ``results/ours'' in the format of ``question id \t the oracle language model ( between the two mentioned) \t predicted language model ''
