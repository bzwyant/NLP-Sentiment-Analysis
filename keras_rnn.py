# Keras imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM


def SentimentLSTM(vocab_size, output_dim, weights, max_seq_length):
	model = Sequential()
	model.add(
		Embedding(input_dim=vocab_size,
		          output_dim=output_dim,
		          weights=[weights],
		          input_length=max_seq_length))
	model.add(Dropout(0.3))
	model.add(LSTM(max_seq_length, return_sequences=True))
	model.add(LSTM(3))
	model.add(Dense(3, activation='softmax'))
	return model
