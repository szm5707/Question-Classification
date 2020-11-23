import torch
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

print("loading the test dataset from TREC_10.label.txt file....")
test_texts, test_labels = readDataset("TREC_10.label.txt")
print("done!\n")

from transformers import DistilBertTokenizerFast
print("downloading the DistilBertTokenizerFast for sentence tokenization...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
print("tokenizer downloaded!\n")

test_encodings = tokenizer(test_texts, truncation=True, padding=True)

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

test_dataset = TRECDataset(test_encodings, test_labels)

from torch.utils.data import DataLoader
from transformers import DistilBertModel, AdamW

print("using CUDA if available, CPU otherwise...\n")
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

print("loading model...")
model.load_state_dict(torch.load("distilBertClassifier.pt", map_location=torch.device('cpu')))
print("model loaded!\n")
model.eval()

criterion = torch.nn.CrossEntropyLoss()

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

print("model is ready for testing!\n")

print("press 1 if you'd like to test the model on the entire dataset")
print("press 2 if you'd like to test the model on your example\n")

print("Press 1 or 2:\n")

val = input("Enter your value: ")

while int(val) not in [1, 2]:
    val = input("Please enter 1 or 2: ")

val = int(val)

if val == 1:
    print("testing on TREC10 dataset....\n")
    test_loss, test_acc = test(test_dataset)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAccuracy: {test_acc * 100:.1f}%(test)')
else:
    input_string = input("Please enter a question now: ")

    enc = tokenizer([input_string], truncation=True, padding=True)

    test_string_data = TRECDataset(enc, [0])

    batch = DataLoader(test_string_data, batch_size=1)

    for i, d in enumerate(batch):

        input_ids = d['input_ids'].to(device, dtype = torch.long)
        attention_mask = d['attention_mask'].to(device, dtype = torch.long)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)

        idx = output.argmax()

        if idx == 0:
            print("DESC")
        elif idx == 1:
            print("ENTY")
        elif idx == 2:
            print("ABBR")
        elif idx == 3:
            print("HUM")
        elif idx == 4:
            print("NUM")
        else: print("LOC")