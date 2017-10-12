from __init__ import *
from keras.models import save_model


class NNModel:
    def __init__(self):
        self.model=None
        self.model=0

    def build_model(self,embeds,passage_maxlength,query_maxlength,DIM,lstm_num,dropout):
        pass

    def train(self,train_data,train_label,valid_data,valid_label,epochs,batch_size):
        if valid_data == None or valid_label == None:
            self.model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size)
        else:
            self.model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size,
                  validation_data=(valid_data, valid_label))

    def predict(self,data):
        res=self.model.predict(data)
        return res

    def save_model(self,filename):
        save_model(self.model,filename)

    def print_model(self):
        print(self.model.summary())

