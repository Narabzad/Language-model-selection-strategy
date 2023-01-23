import json, math
from sentence_transformers import InputExample, CrossEncoder, losses,  SentenceTransformer, util
from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator


from sentence_transformers import evaluation
f = open('BioASQ/train_7b.tsv','r').readlines() 
label2int = {"no": 0, "yes": 1}
train_samples=[]
ids=[]

for line  in f:
    id,question,context,answer=line.rstrip().split('\t')
    train_samples.append(InputExample(texts=[question, context], label=label2int[answer]))
    ids.append(id)
f = open('BioASQ/test_7b.tsv','r').readlines() 
test_samples=[]
test_labels=[]
test_labels_input=[]
for line  in f:
    id,question,context,answer=line.rstrip().split('\t')
    test_samples.append([question, context])
    test_labels.append(label2int[answer])
    test_labels_input.append((InputExample(texts=[question, context], label=label2int[answer])))
    
f = open('BioASQ/dev_7b.tsv','r').readlines() 
dev_labels_input=[]
for line  in f:
    id,question,context,answer=line.rstrip().split('\t')
    dev_labels_input.append((InputExample(texts=[question, context], label=label2int[answer])))
    
    
print(len(test_labels))
model_transformers={'distilbert':'distilbert-base-uncased','bert':'bert-base-uncased',
                    'distilroberta':'distilroberta-base','roberta':'roberta-base','pubmed':'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'}

num_epochs=10
for transformer_name in ['distilroberta','distilbert','bert','roberta','pubmed']:
    print(transformer_name,num_epochs)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

    model = CrossEncoder(model_transformers[transformer_name], num_labels=1,max_length=512)
    model_name='models/fine-tuned/'+transformer_name.split('/')[-1]
    # Train the model
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_labels_input, name=model_name.split('/')[-1],write_csv=True)

    model.fit(train_dataloader=train_dataloader,
            epochs=num_epochs,
            warmup_steps=warmup_steps,          evaluator=evaluator, output_path=model_name, save_best_model=True
            )

    scores=model.predict(test_samples)

    out=open('results/fine-tuned-standalone/'+model_name.split('/')[-1]+'.test.tsv','w')
    for i in range(len(test_samples)):
        out.write(ids[i]+'\t'+str(test_labels[i])+'\t'+str(scores[i])+'\n')
    out.close()
