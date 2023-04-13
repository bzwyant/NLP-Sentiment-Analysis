import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class SentimentRNN(nn.Module):
	def __init__(self, word_embeddings, embedding_dim: int, hidden_dim: int,
	             output_dim: int, num_layers: int = 2, dropout: float = 0.2):
		
		super(SentimentRNN, self).__init__()
		embeddings = torch.from_numpy(word_embeddings).float()
		# embedding layer to map input indices to embedding vectors
		self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)  # don't update word embeddings as it trains
		
		# LSTM layer
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
		
		# dropout layer
		self.dropout = nn.Dropout(dropout)
		
		# Linear layer used for output
		self.fc = nn.Linear(hidden_dim, output_dim)
		
		# sigmoid layer to classify
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, token):
		
		# convert so types match
		token = token.long()
		
		# convert the input token to its word embedding
		embedded = self.embedding(token)
	
		# pass the embedded input to the LSTM
		output, hidden = self.lstm(embedded)
	
		# get the last output
		output = output[:, -1, :]
		
		# apply dropout to prevent over fitting
		output = self.dropout(output)
		
		# pass to the fully connected layer
		output = self.fc(output)
		
		# apply sigmoid to classify
		output = self.sigmoid(output)
	
		return output
		
		

# https://www.kaggle.com/code/arunmohan003/sentiment-analysis-using-lstm-pytorch
# https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L15/1_lstm.ipynb
# https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html?highlight=from_pretrained#torch.nn.Embedding.from_pretrained
# https://colab.research.google.com/github/agungsantoso/deep-learning-v2-pytorch/blob/master/sentiment-rnn/Sentiment_RNN_Exercise.ipynb#scrollTo=qoHTVsZlpmwC