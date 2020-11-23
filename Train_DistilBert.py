# -*- coding: utf-8 -*-
"""
In this work, I finetune DistilBERT to classify the TREC Dataset. DistilBERT is a distilled or approximate version of BERT. It retains about 97% of BERT's performance while using only half the number of parameters. This property enables DistilBERT to be a small, fast, cheap and light Transformer model for efficient training. 

##In order to begin traing, first upload the "train_5500.label.txt" and "TREC_10.label.txt" files in the current working directory.
"""

#reading the dataset
from pathlib import Path

categories = {
    "DESC": 0,
    "ENTY": 1,
    "ABBR": 2,
    "HUM": 3,
    "NUM": 4,
    "LOC": 5
}

def readDataset(path):
    
    texts = []
    labels = []
    
    f = open(path, "r", encoding = "ISO-8859-1")
    
    for line in f:
        
        line = line.rstrip().split(" ", 1)
        text = line[1]
        texts.append(text)
        
        label = line[0].split(":")[0]
        labels.append(categories[label])
    
    return texts, labels

train_texts, train_labels = readDataset("train_5500.label.txt")
test_texts, test_labels = readDataset("TREC_10.label.txt")

"""Using 20% of the training set for validation."""

#using a 60-40 split for training and validation
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.4)

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

"""In order to feed batches of input during training, all inputs are truncated to the model's maximum input length and are padded to be of the same length."""

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

import torch

class TRECDataset(torch.utils.data.Dataset):

  def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

  def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

  def __len__(self):
        return len(self.labels)

train_dataset = TRECDataset(train_encodings, train_labels)
val_dataset = TRECDataset(val_encodings, val_labels)
test_dataset = TRECDataset(test_encodings, test_labels)

#defining my model. I add and train a droput and a final linear layer to get an 
#output vector of 6.
from torch.utils.data import DataLoader
from transformers import DistilBertModel, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 6)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = DistillBERTClass()
model.to(device)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

#using the CrossEntropyLoss function and AdamW optimizer for training our model.
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

def train(train_data):

  train_loss = 0.0
  train_acc = 0

  batch = DataLoader(train_data, batch_size=16, shuffle=True)

  for i, d in enumerate(batch):

    input_ids = d['input_ids'].to(device, dtype = torch.long)
    attention_mask = d['attention_mask'].to(device, dtype = torch.long)
    labels = d['labels'].to(device, dtype = torch.long)

    optimizer.zero_grad()

    output = model(input_ids, attention_mask=attention_mask)

    loss = criterion(output, labels)

    train_loss += loss.item()

    loss.backward()
    optimizer.step()

    train_acc += (output.argmax(1) == labels).sum().item()

  
  return train_loss/len(train_data), train_acc/len(train_data)

def test(data):

  loss = 0.0
  acc = 0

  batch = DataLoader(data, batch_size=16, shuffle=True)

  for i, d in enumerate(batch):

    input_ids = d['input_ids'].to(device, dtype = torch.long)
    attention_mask = d['attention_mask'].to(device, dtype = torch.long)
    labels = d['labels'].to(device, dtype = torch.long)

    with torch.no_grad():

      output = model(input_ids, attention_mask=attention_mask)
      loss += criterion(output, labels)
      loss == loss.item()

      acc += (output.argmax(1) == labels).sum().item()
    
  return loss/len(data), acc/len(data)

for epoch in range(5):

  train_loss, train_acc = train(train_dataset)
  val_loss, val_acc = test(val_dataset)

  print(f'\tLoss: {train_loss:.4f}(train)\t|\tAccuracy: {train_acc * 100:.1f}%(train)')
  print(f'\tLoss: {val_loss:.4f}(validation)\t|\tAccuracy: {val_acc * 100:.1f}%(validation)')

test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAccuracy: {test_acc * 100:.1f}%(test)')

torch.save(model.state_dict(), "distilBertClassifier.pt") #saving the model
#this model can be downloaded and used for inference using the Test_DistilBert.py
#file