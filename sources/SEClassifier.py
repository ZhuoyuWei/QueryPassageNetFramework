from __init__ import *
from NNModel import NNModel
from utils.multi_gpu import make_parallel


class SEClassifier(NNModel):
    def build_model(self,embeds,passage_maxlength,query_maxlength,DIM,lstm_num,dropout):

        passage = Input(shape=(passage_maxlength,), dtype='int32')
        passage_embedding = Embedding(len(embeds),
                                      len(embeds[0]),
                                      weights=[embeds],
                                      input_length=passage_maxlength,
                                      trainable=False)(passage)
        # passage_lstm_layers = LSTM(50, return_sequences=True)(passage_embedding)
        passage_lstm_layers = passage_embedding
        for i in range(lstm_num):
            passage_lstm_layers = LSTM(DIM, return_sequences=True)(passage_lstm_layers)
            if dropout:
                passage_lstm_layers = Dropout(0.5)(passage_lstm_layers)

        query = Input(shape=(query_maxlength,), dtype='int32')
        query_embedding = Embedding(len(embeds),
                                    len(embeds[0]),
                                    weights=[embeds],
                                    input_length=query_maxlength,
                                    trainable=False)(query)

        query_lstm_layers = query_embedding
        for i in range(lstm_num - 1):
            query_lstm_layers = LSTM(DIM, return_sequences=True)(query_lstm_layers)
            if dropout:
                query_lstm_layers = Dropout(0.5)(query_lstm_layers)

        query_lstm_layers = LSTM(DIM, return_sequences=False)(query_lstm_layers)

        query_lstm_repeat = RepeatVector(passage_maxlength)(query_lstm_layers)

        # query_passage_dot=TimeDistributed(Dot(2))([query_lstm_repeat,passage_lstm_layers])

        query_passage_mul = Multiply()([query_lstm_repeat, passage_lstm_layers])
        query_passage_dot = Lambda(lambda x: K.sum(x, axis=2))(query_passage_mul)

        # query_passage_dot_rep = RepeatVector(DIM)(query_passage_dot)
        # query_passage_dot_rep_trans = Lambda(transpose021_fun, output_shape=(passage_maxlength, DIM))(query_passage_dot_rep)
        # attended_layer = Multiply()([passage_lstm_layers, query_passage_dot_rep_trans])

        start_point = Dense(passage_maxlength, activation='softmax')(query_passage_dot)
        end_point = Dense(passage_maxlength, activation='softmax')(start_point)

        # start_point = Dense(1, activation='softmax')(attended_layer)
        # end_point=Dense(1, activation='softmax')(attended_layer)

        # start_point_reshape=Reshape((passage_maxlength,), input_shape=(passage_maxlength,1,))(start_point)
        # end_point_reshape=Reshape((passage_maxlength,), input_shape=(passage_maxlength,1,))(end_point)

        model = Model(inputs=[passage, query], outputs=[start_point, end_point])
        model=make_parallel(model,4)


        model.compile(optimizer='adagrad',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self.model=model
        print(model.summary())

    def predict(self,data):
        res=self.model.predict(data)
        res_index = np.argmax(res, axis=2)
        res_index=np.transpose(res_index)
        return res_index

    def reshape_data(self,data):
        return data[:2],data[2:]
