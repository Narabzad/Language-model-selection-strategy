from sentence_transformers import InputExample, CrossEncoder, losses,  SentenceTransformer, util
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import os 

label2int = {"no": 0, "yes": 1}
f = open('BioASQ/test_7b.tsv','r').readlines() 
test_samples=[]
test_labels=[]
test_labels_input=[]
ids=[]
for line  in f:
    id,question,context,answer=line.rstrip().split('\t')
    ids.append(id)
    test_samples.append([question, context])
    test_labels.append(label2int[answer])
    test_labels_input.append((InputExample(texts=[question, context], label=label2int[answer])))

for classifier in os.listdir('models/classifier/'):
    try:
        modelname1=classifier.split('-vs-')[0]
        modelname2=classifier.split('-vs-')[1]
        
        model = CrossEncoder('models/classifier/'+classifier , num_labels=2,max_length=512)
        f1_dic=[]
        f1_file=open('results/fine-tuned-standalone/'+modelname1+'.test.tsv','r').readlines()
        for line in f1_file:
            id,label,prediction=line.split('\t')
            f1_dic.append(float(prediction))

        f2_dic=[]
        f2_file=open('results/fine-tuned-standalone/'+modelname2+'.test.tsv','r').readlines()
        for line in f2_file:
            id,label,prediction=line.split('\t')
            f2_dic.append(float(prediction))

        scores=model.predict(test_samples)
        out=open('results/ours/'+classifier+'.test.tsv','w')
        for i in range(len(test_samples)):
            if scores[i][0]>scores[i][1] : 
                out.write(ids[i]+'\t'+str(test_labels[i])+'\t'+str(f2_dic[i])+'\n')
            else:
                out.write(ids[i]+'\t'+str(test_labels[i])+'\t'+str(f1_dic[i])+'\n')
        out.close()
    except:
        pass
