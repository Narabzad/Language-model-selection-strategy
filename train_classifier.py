from sentence_transformers import InputExample, CrossEncoder, losses,  SentenceTransformer, util
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import os 

f = open('BioASQ/train_8b.tsv','r').readlines() 
label2int = {"no": 0, "yes": 1}
train_samples=[]
labels=[]
for line  in f:
    id,question,context,answer=line.rstrip().split('\t')
    train_samples.append([question, context])
    labels.append(int(answer))
    

model_transformers={'distilbert':'distilbert-base-uncased','bert':'bert-base-uncased',
                    'distilroberta':'distilroberta-base','roberta':'roberta-base','pubmed':'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'}
for model_name1 in  os.listdir('models/fine-tuned'):
    for model_name2 in   os.listdir('models/fine-tuned'):
        if model_name2 == model_name1:
            continue
        
        model = CrossEncoder('models/fine-tuned/'+model_name1, num_labels=1,max_length=512)
        scores1=model.predict(train_samples)
        model = CrossEncoder('models/fine-tuned/'+model_name2, num_labels=1,max_length=512)
        scores2=model.predict(train_samples)

        train_classifier_samples=[]
        for i in range(len(train_samples)):
            
            if abs(labels[i] - scores1[i]) < abs(labels[i]-scores2[i]):
                train_classifier_samples.append(InputExample(texts=train_samples[i],label=0))
            else:
                train_classifier_samples.append(InputExample(texts=train_samples[i],label=1))

        train_dataloader = DataLoader(train_classifier_samples, shuffle=True, batch_size=16)
        warmup_steps = 100

        model = CrossEncoder(model_transformers[model_name1], num_labels=2,max_length=512)


        model.fit(train_dataloader=train_dataloader,
                epochs=10,
                warmup_steps=warmup_steps, output_path='models/classifier/'+model_name1+'-vs-'+model_name2, save_best_model=True
                )
        model.save('models/classifier/'+model_name1+'-vs-'+model_name2)

