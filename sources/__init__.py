
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout,Dot
from keras.layers import Conv1D, MaxPooling1D, Embedding,LSTM,Activation,Multiply,Reshape
from keras.models import Model
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.activations import tanh, softmax
from keras.layers import Input, LSTM, Dense, RepeatVector, Lambda, Activation
from keras.layers.wrappers import TimeDistributed
from keras.engine import InputSpec
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback, LearningRateScheduler
from keras.initializers import Initializer
from keras import initializers
from keras.layers.recurrent import _time_distributed_dense

import nltk
import numpy as np