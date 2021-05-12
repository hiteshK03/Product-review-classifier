import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import re
import os
import time
from transformers import AutoTokenizer, AutoModel, AdamW
import lime
import torch
import torch.nn.functional as F
import matplotlib
from lime.lime_text import LimeTextExplainer

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# reviews_train = pd.read_csv('../train.csv')
reviews_val = pd.read_csv('../gold_test.csv')

# train_texts = reviews_train.reviews
# train_labels = reviews_train.ratings
val_texts = reviews_val.reviews
val_labels = reviews_val.ratings



texts = []
for i in val_texts:
    texts.append(i)

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

def predictor(texts):
	saved_model = torch.load('26_model.pt', map_location=torch.device('cpu'))
	tokens = tokenizer(texts, return_tensors='pt', padding=True)
	output = saved_model(tokens['input_ids'].to(device), tokens['attention_mask'].to(device))
	tensor_logits = output
	probas = F.softmax(tensor_logits).cpu().detach().numpy()
	return probas

def lime_explainer(text):
	probs = predictor(text)
	pred = np.argmax(probs, axis=1)
	explainer = LimeTextExplainer(class_names=['1','2','3','4','5'])
	exp = explainer.explain_instance(text, predictor, (0,1,2,3,4,), num_features=10, num_samples=1000)
	print("Predicted Label : ", pred)
	fig = exp.as_pyplot_figure(label=pred[0])
	fig.savefig("lime_report.jpg")
	exp.save_to_file('lime_report.html')
	return probs	


# idx = 54
# text = val_texts[idx]
# label = val_labels[idx]
# explainer = LimeTextExplainer(class_names=['1','2','3','4','5'])
# exp = explainer.explain_instance(text, predictor, (label-1,), num_features=10, num_samples=1000)
# print("True Label : ", label)
# fig = exp.as_pyplot_figure(label=label-1)
# fig.savefig("lime_report.jpg")
# exp.save_to_file('lime_report.html')