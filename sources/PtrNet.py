from __init__ import *
from NNModel import NNModel
from keras.layers.recurrent import _time_distributed_dense

def transpose021_fun(x):
    return K.permute_dimensions(x, [0,2,1])

class FirstLayer(Layer):
    def __init__(self, **kwargs):
        super(FirstLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        pass

    def call(self, inputs):
        return inputs[:, :2]

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], 2, input_shape[1])
        return output_shape

class PointerLSTM(LSTM):
    def __init__(self, hidden_shape, *args, **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        super(PointerLSTM, self).__init__(units=hidden_shape,*args, **kwargs)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        '''
        init = initializers.get('orthogonal')
        self.W1 = init((self.hidden_shape, 1))
        self.W2 = init((self.hidden_shape, 1))
        self.vt = init((input_shape[1], 1))
        self.trainable_weights += [self.W1, self.W2, self.vt]
        '''

        self.W1 = self.add_weight(shape=(input_shape[2],1),
                                    name='w1',
                                    initializer='orthogonal')

        self.W2 = self.add_weight(shape=(self.hidden_shape, 1),
                              name='w2',
                              initializer='orthogonal')

        self.vt = self.add_weight(shape=(input_shape[1], 1),
                              name='vt',
                              initializer='orthogonal')

    '''
    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1] - 1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_state(x_input)

        constants = super(PointerLSTM, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs
    '''
    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_state = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        timesteps = input_shape[1]
        if self.unroll and timesteps in [None, 1]:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined or equal to 1. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        #tmp_inputs=
        constants.append(inputs)

        preprocessed_input = self.preprocess_input(inputs, training=None)
        #print('&&&&&&&&&&&&&&&&&&&&&&&&&'+str(inputs.shape)+'\t'+str(input_shape)+'\t'+str(preprocessed_input.shape))
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_state,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output


    def step(self, x_input, states):
        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states[:-1])

        print('****************************************'+str(en_seq.shape))

        en_shape = K.int_shape(en_seq)
        en_input_dim = en_shape[2]
        en_timesteps = en_shape[1]

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        print('----------------------------------------'+str(dec_seq.shape))
        Eij = _time_distributed_dense(en_seq, self.W1,input_dim=en_input_dim,timesteps=en_timesteps,output_dim=1)
        Dij = _time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    #def get_output_shape_for(self, input_shape):
    def compute_output_shape(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

class PtrNet(NNModel):
    def build_model(self, embeds, passage_maxlength, query_maxlength, DIM, lstm_num, dropout):

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

        query_passage_dot_rep = RepeatVector(DIM)(query_passage_dot)
        query_passage_dot_rep_trans = Lambda(transpose021_fun, output_shape=(passage_maxlength, DIM))(
            query_passage_dot_rep)

        attended_layer = Multiply()([passage_lstm_layers, query_passage_dot_rep_trans])
        attended_layer = Dense(passage_maxlength, activation="softmax")(attended_layer)

        pointer_net = PointerLSTM(passage_maxlength, return_sequences=True)(attended_layer)

        # output1=Dense(maxlength,activation='softmax')(FirstLayer()(pointer_net[:,0]))
        # output2=Dense(maxlength,activation='softmax')(FirstLayer()(pointer_net[:,1]))

        pointer_net = FirstLayer()(pointer_net)
        dense_layer2 = TimeDistributed(Dense(passage_maxlength, activation='softmax'))(pointer_net)

        # output1=K.argmax(pointer_net,axis=)

        # model = Model(inputs=inputs, outputs=pointer_net)





        # model = Model(inputs=[passage,query], outputs=[start_point, end_point])
        model = Model(inputs=[passage, query], outputs=dense_layer2)
        model.compile(optimizer='sgd',  # rmsprop
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def predict(self,data):
        res=self.model.predict(data)
        res_index = np.argmax(res, axis=2)
        return res_index