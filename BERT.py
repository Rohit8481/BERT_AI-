import pandas as pd 
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import AdamW
content = pd.read_csv("your_data_file_name_here.csv")
content = content.dropna(subset=['label'])
content['label'] = content['label'].astype(int)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2,
                                                      problem_type="single_label_classification" # for binary class
                                                      )
optimizer = AdamW(model.parameters(), lr=2e-5)

embedding = tokenizer(content["body"].tolist(),
                      padding = True, # for equal list values, ex : ['hello world'] -> [101 2332 3421 0 0 102], ['hello i am rohit'] -> [101 2322 1221 4321 6332 102]
                      truncation = True, # it handles long text
                      max_length = 100,
                      return_tensors = 'pt')

labels = torch.tensor(content['label'].values, dtype=torch.long)

#################### training part ################################

class Custom_dataset(Dataset) : 
    
    def __init__(self,labels, embedding):
        self.embedding = embedding
        self.labels = labels
    
    def __getitem__(self, index):
        input_ids = self.embedding["input_ids"][index]
        attentions = self.embedding["attention_mask"][index]
        label = self.labels[index]

        return {
        "input_ids": input_ids,
        "attention_mask": attentions,
        "labels": label
    }
    
    def __len__(self):
            return len(self.labels)

loader = DataLoader(Custom_dataset(labels, embedding), # to load the data
                    batch_size = 8,
                    shuffle = True
                    )

device = (torch.device("cpu")) # to use the CPU for the training, we can also use the GPU by "cuda" but i dont want to use GPU 
model.to(device)

echo = 3

for i in range(echo):
    model.train()
    total_loss = 0
    
    for j in tqdm(loader):
        optimizer.zero_grad() # clear all previous gradients

        input_ids = j['input_ids'].to(device)              # transfer tensors to the device such as CPU or GPU  ( the device should be same if CPU so tensors also transfer to the CPU instead of GPU )
        attentions = j['attention_mask'].to(device)        # transfer tensors to the device such as CPU or GPU ( the device should be same if CPU so tensors also transfer to the CPU instead of GPU )
        labels = j['labels'].to(device)                    # transfer tensors to the device such as CPU or GPU ( the device should be same if CPU so tensors also transfer to the CPU instead of GPU )

        output = model(input_ids = input_ids,
                    attention_mask = attentions ,
                    labels = labels
                    )
        loss = output.loss
        total_loss += loss.item()
        print(total_loss)

        loss.backward() # Back propagation
        optimizer.step() # optimize weights

model.save_pretrained('my_email_phishing_detector')
tokenizer.save_pretrained('my_email_phishing_detector')

