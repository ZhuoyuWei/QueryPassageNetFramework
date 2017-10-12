from __init__ import *
from PtrNet import PtrNet
from SEClassifier import SEClassifier
from NNModel import NNModel
import predata

import sys
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__=="__main__":

    start=time.time()
    word2id, embeds, DIM = predata.read_embeddings_from_txt(sys.argv[1])
    end = time.time()
    print('read emb time: '+str(end-start))


    #for 10000 little dataset
    #data,passage_maxlength,query_maxlength=predata.read_data_from_file(sys.argv[2],word2id)
    #train_data,valid_data=predata.SimplySplitData(data,13347)

    #for big data
    start=time.time()
    train_data,train_p_maxlength,train_q_maxlength=predata.read_data_from_file_simply(sys.argv[2])
    valid_data,valid_p_maxlength,valid_q_maxlength=predata.read_data_from_file_simply(sys.argv[3])
    passage_maxlength=train_p_maxlength if train_p_maxlength>valid_p_maxlength else valid_p_maxlength
    query_maxlength=train_q_maxlength if train_q_maxlength>valid_q_maxlength else valid_q_maxlength
    train_data=predata.ReshapeForNN(train_data,query_maxlength,passage_maxlength,word2id)
    valid_data=predata.ReshapeForNN(valid_data,query_maxlength,passage_maxlength,word2id)


    end=time.time()
    print('read data time: '+str(end-start))

    start=time.time()
    nnmodel=SEClassifier()
    nnmodel.build_model(embeds,passage_maxlength,query_maxlength,DIM,1,False)
    end=time.time()
    print('modeling time: '+str(end-start))

    start=time.time()
    nnmodel.train(train_data[:2],train_data[2:],valid_data[:2],valid_data[2:],300,4096)
    end=time.time()
    print('training time: '+str(end-start))
    if not sys.argv[5] == None:
        nnmodel.save_model(sys.argv[5])

    start=time.time()
    res=nnmodel.predict(valid_data[:2])
    end=time.time()
    print('predict time: '+str(end-start))

    predata.Eva(res,valid_data[2:])

    if not sys.argv[4] == None:
        predata.print_valid_res(res,sys.argv[4])






