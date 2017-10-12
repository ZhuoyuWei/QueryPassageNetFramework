from __init__ import *
from PtrNet import PtrNet
from SEClassifier import SEClassifier
from NNModel import NNModel
from PtrSimple import PtrSimple
import predata

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

if __name__=="__main__":
    word2id, embeds, DIM = predata.read_embeddings_from_txt(sys.argv[1])

    #for 10000 little dataset
    data,passage_maxlength,query_maxlength=predata.read_data_from_file(sys.argv[2],word2id)
    train_data,valid_data=predata.SimplySplitData(data,13347)

    #for big data
    #train_data,train_p_maxlength,train_q_maxlength=predata.read_data_from_file_simply(sys.argv[2],word2id)
    #valid_data_data,valid_p_maxlength,valid_q_maxlength=predata.read_data_from_file_simply(sys.argv[3],word2id)
    #passage_maxlength=train_p_maxlength if train_p_maxlength>valid_p_maxlength else valid_p_maxlength
    #query_maxlength=train_q_maxlength if train_q_maxlength>valid_q_maxlength else valid_q_maxlength

    #nnmodel=PtrSimple()
    nnmodel=SEClassifier()

    nnmodel.build_model(embeds,passage_maxlength,query_maxlength,DIM,1,False)
    nnmodel.print_model()
    trainx,trainy=nnmodel.reshape_data(train_data)
    validx,validy=nnmodel.reshape_data(valid_data)

    nnmodel.train(trainx,trainy,validx,validy,500,1024)

    res=nnmodel.predict(valid_data[:2])
    predata.Eva(res,valid_data[2:])

    if not sys.argv[3] == None:
        predata.print_valid_res(res,sys.argv[3])





