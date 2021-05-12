import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import re
import os
import time
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AdamW

class NLPDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        item['label'] = torch.tensor(self.labels[idx]-1)
        item['input_ids'] = torch.tensor(item['input_ids'])
        item['attention_mask'] = torch.tensor(item['attention_mask'])
        return item

class CustomBERTModel(torch.nn.Module):
    def __init__(self, num_classes):
          super(CustomBERTModel, self).__init__()
          self.bert = AutoModel.from_pretrained("distilbert-base-uncased")

          ### New layers:
          self.linear1 = torch.nn.Linear(768, 256)
          self.linear2 = torch.nn.Linear(256, num_classes) ## as you have 4 classes in the output
          # self.sig = torch.nn.functional.sigmoid()

    def forward(self, ids, mask):
          sequence_output = self.bert(ids,attention_mask=mask)
#           print(sequence_output["last_hidden_state"].shape)
          sequence_output = sequence_output["last_hidden_state"][:,0,:]

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output.view(-1,768))
          linear2_output = self.linear2(linear1_output)
          # linear2_output = self.sig(linear2_output)

          return linear2_output


def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    return text.lower()

def remove_punctuation(text):
    # return the reviews after removing punctuations
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    # text = re.sub(r'[^\w\s]',' ',text)
    text = re.sub(' +', ' ',text)
    return text

def remove_stopwords(text):
    # return the reviews after removing the stopwords
    exclude = set(['$','&','+',':',';','=','@','|','<','>','^','*','%','-','#','\'','।'])
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    return ''.join(ch for ch in text if ch not in exclude)
    # stop_words = stopwords.words('english')
    # stop_words.remove('not')
    # return ' '.join([w for w in text if not w in stop_words])  

def perform_tokenization(text):
    # return the reviews after performing tokenization
    return word_tokenize(text)

def perform_padding(data, TEXT):
    # return the reviews after padding the reviews to maximum length
    # return pad_sequences(data, maxlen=max_len)
    return TEXT.pad(data)

def preprocess_review(text):
    out = convert_to_lower(text)
    # out = remove_punctuation(out)
    # out = remove_stopwords(out)
    # out = perform_tokenization(out)
    out = tokenizer(out, padding='max_length', truncation=True)
    return out

def preprocess_data(data):
    # make all the following function calls on your data
    # EXAMPLE:->

    review = data.apply(lambda row: preprocess_review(row))
    # review, text_field = encode_data(review, L)
    return review

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

reviews_train = pd.read_csv('train.csv')
reviews_val = pd.read_csv('gold_test.csv')
reviews_test = pd.read_csv('test.csv')

train_texts = reviews_train.reviews
train_labels = reviews_train.ratings
val_texts = reviews_val.reviews
val_labels = reviews_val.ratings

class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
print(class_weights)
weights = [class_weights[train_labels[i]-1] for i in range(int(num_samples))]
weight_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

train_encodings = preprocess_data(train_texts)
val_encodings = preprocess_data(val_texts)

train_dataset = NLPDataset(train_encodings, train_labels)
val_dataset = NLPDataset(val_encodings, val_labels)

train_length = len(train_dataset)
val_length = len(val_dataset)

model = CustomBERTModel(num_classes=5)
for param in model.bert.parameters():
    param.requires_grad = False
model.to(device)
model.train()

# validation_split = .4
# shuffle_dataset = True
# random_seed= 42

# batch_size = 200
# # train_set = torch.utils.data.TensorDataset(self.reviews, self.ratings)
# dataset_size = len(train_dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
#                                         sampler=train_sampler, drop_last=True)
# val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
#                                         sampler=valid_sampler, drop_last=False)
# train_length = dataset_size - split
# val_length = split

train_loader = DataLoader(train_dataset, batch_size=250, sampler=weight_sampler)
val_loader = DataLoader(val_dataset, batch_size=250, shuffle=True, drop_last=True)
criterion = torch.nn.CrossEntropyLoss()
optim = AdamW(model.parameters(), lr=5e-4)
total_epochs = 20

history = []

for epoch in range(total_epochs):
    trainLoss = 0.0
    trainAcc = 0.0

    validLoss = 0.0
    validAcc = 0.0
    
    epochStart = time.time()
    print("Epoch: {}/{}".format(epoch+1, total_epochs))
    
    for i, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)        
        outputs = model(input_ids, mask = attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
        trainLoss += loss.item() * input_ids.size(0)
        _, predictions = torch.max(outputs.data, 1)
        corrCounts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(corrCounts.type(torch.FloatTensor))
        trainAcc += acc.item() * input_ids.size(0)
        if i%20==0:
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(input_ids, mask = attention_mask)

            # Compute loss
            loss = criterion(outputs, labels)

            # Compute the total loss for the batch and add it to validLoss
            validLoss += loss.item() * input_ids.size(0)

            # Calculate validation accuracy
            _, predictions = torch.max(outputs.data, 1)
            corrCounts = predictions.eq(labels.data.view_as(predictions))

            # Convert corrCounts to float and then compute the mean
            acc = torch.mean(corrCounts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to validAcc
            validAcc += acc.item() * input_ids.size(0)
            if j%10==0:
                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
    model_path = os.path.join('./', str(epoch)+'_model.pt')
    torch.save(model, model_path)
    print('Saved Model checkpoints')

    # Find average training loss and training accuracy
    trainLossAvg = trainLoss/train_length
    trainAccAvg = trainAcc/train_length

    # Find average training loss and training accuracy
    validLossAvg = validLoss/val_length
    validAccAvg = validAcc/val_length

    history.append([trainLossAvg, validLossAvg, trainAccAvg, validAccAvg])

    epochEnd = time.time()

    print("Epoch : {:03d}, Training: Loss : {:.4f}, Accuracy: {:.4f}%".format(epoch, trainLossAvg, trainAccAvg*100))
    print("Validation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(validLossAvg, validAccAvg*100, epochEnd-epochStart))

##################################################################
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

saved_model = model
# saved_model = torch.load('./23_model.pt')

test_acc = 0.0
test_loss = 0.0
test_preds = []

# Validation - No gradient tracking needed
with torch.no_grad():

    # Set to evaluation mode
    saved_model.eval()

    # Validation loop
    for j, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = saved_model(input_ids, mask = attention_mask)

        _, predictions = torch.max(outputs.data, 1)
        test_preds.append(predictions[0].item()+1)

from sklearn.metrics import classification_report
test_preds = np.array(test_preds)
print(classification_report(val_labels, test_preds))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(val_labels, test_preds))

##################################################################

import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

texts = []
for i in val_texts:
    texts.append(i)

def predictor(texts):
    tokens = tokenizer(texts, return_tensors='pt', padding=True)
    output = model(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))
    tensor_logits = output
    probas = F.softmax(tensor_logits).cpu().detach().numpy()
    return probas

idx = 54
text = train_texts[idx]
label = train_labels[idx]
explainer = LimeTextExplainer(class_names=['1','2','3','4','5'])
exp = explainer.explain_instance(text, predictor, (label-1,), num_features=10, num_samples=1000)
print("True Label : ", label)
exp.show_in_notebook(text=text)