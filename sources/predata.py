from __init__ import *

def read_embeddings_from_txt(filename):
    word2vec={}
    embeds=[]
    count=int(0)
    error_ems=0
    flag=False
    with open(filename) as f:
        for line in f:
            ss=line.strip().split()
            word2vec[ss[0]]=count
            if flag and not len(ss) == dim+1:
                error_ems+=1
                continue
            ss=np.asarray(ss[1:], dtype='float32')
            embeds.append(ss)
            count+=1
            dim=len(ss)
            flag=True
    print('dim: '+str(dim)+'\terrors: '+str(error_ems))

    word2vec['<e1>'] = count
    word2vec['<\\e1>'] = count + 1
    word2vec['<e2>'] = count + 2
    word2vec['<\\e2>'] = count + 3
    word2vec['UNK'] = count + 4
    word2vec['BLANK'] = count + 5

    for i in range(5):
        embeds.append(np.random.rand(1, dim))
    embeds.append(np.zeros((1, dim), dtype='float32'))

    #embeds=np.matrix(embeds)
    #embeds=np.array(embeds)

    tmp=embeds
    embeds=np.zeros((len(embeds),dim))
    for i in range(len(tmp)):
        embeds[i]=tmp[i]


    return word2vec,embeds,dim

def read_data_from_file(filename,word2id):
    query_maxlength=0
    passage_maxlength=0
    plist=[]
    qlist=[]
    slist=[]
    elist=[]
    selist=[]
    with open(filename) as fin:
        for line in fin:
            ss=line.strip().split('\t')
            if len(ss) == 6:
                query=ss[0]+' '+ss[1]
                answer=ss[2]
                passage=ss[3]

                query=query.lower()
                passage=passage.lower()
                answer=answer.lower()

                query_token=nltk.word_tokenize(query)
                passage_token=nltk.word_tokenize(passage)
                answer_token=nltk.word_tokenize(answer)

                start=-1
                end=-1
                for i in range(len(passage_token)):
                    if answer_token[0] == passage_token[i]:
                        flag=True
                        for j in range(1,len(answer_token)):
                            if not answer_token[j] == passage_token[i+j]:
                                flag=False
                                break

                        if flag:
                            start=i
                            end=i+len(answer_token)

                if start>0 and end>0:
                    #pqlist.append([passage_token,query_token])
                    #selist.append([start,end])
                    plist.append(passage_token)
                    qlist.append(query_token)
                    slist.append(start)
                    elist.append(end)
                    if len(passage_token) > passage_maxlength:
                        passage_maxlength=len(passage_token)
                    if len(query_token) > query_maxlength:
                        query_maxlength=len(query_token)

    print('passage max length is '+str(passage_maxlength))
    print('query max length is'+str(query_maxlength))



    plist_index = np.zeros((len(plist), passage_maxlength))
    for i in range(len(plist)):
        for j in range(len(plist[i])):
            if plist[i][j] in word2id:
                plist_index[i][j]=word2id[plist[i][j]]
            else:
                plist_index[i][j] = word2id['UNK']

    qlist_index = np.zeros((len(qlist), query_maxlength))
    for i in range(len(qlist)):
        for j in range(len(qlist[i])):
            if qlist[i][j] in word2id:
                qlist_index[i][j]=word2id[qlist[i][j]]
            else:
                qlist_index[i][j] = word2id['UNK']

    slist_index=np.zeros((len(slist),passage_maxlength))
    for i in range(len(slist)):
        slist_index[i][slist[i]]=1

    elist_index=np.zeros((len(elist),passage_maxlength))
    for i in range(len(elist)):
        elist_index[i][elist[i]]=1


    return [plist_index,qlist_index,slist_index,elist_index],passage_maxlength,query_maxlength

def read_data_from_file_simply(filename):
    query_maxlength=0
    passage_maxlength=0
    plist=[]
    qlist=[]
    slist=[]
    elist=[]
    with open(filename) as fin:
        for line in fin:
            ss=line.strip().split('\t')
            if len(ss) == 6:
                query=ss[0]+' '+ss[1]
                answer=ss[2]
                passage=ss[3]

                query=query.lower().strip()
                passage=passage.lower()
                answer=answer.lower()

                query_token=nltk.word_tokenize(query)
                passage_token=nltk.word_tokenize(passage)
                answer_token=nltk.word_tokenize(answer)

                start=int(ss[4])
                end=int(ss[5])

                if start>=0 and end>=0:
                    #pqlist.append([passage_token,query_token])
                    #selist.append([start,end])
                    plist.append(passage_token)
                    qlist.append(query_token)
                    slist.append(start)
                    elist.append(end)
                    if len(passage_token) > passage_maxlength:
                        passage_maxlength=len(passage_token)
                    if len(query_token) > query_maxlength:
                        query_maxlength=len(query_token)

    print('passage max length is '+str(passage_maxlength))
    print('query max length is'+str(query_maxlength))


    return [plist,qlist,slist,elist],passage_maxlength,query_maxlength

def ReshapeForNN(data,query_maxlength,passage_maxlength,word2id):

    plist=data[0]
    qlist=data[1]
    slist=data[2]
    elist=data[3]

    plist_index = np.zeros((len(plist), passage_maxlength))
    for i in range(len(plist)):
        for j in range(len(plist[i])):
            if plist[i][j] in word2id:
                plist_index[i][j]=word2id[plist[i][j]]
            else:
                plist_index[i][j] = word2id['UNK']

    qlist_index = np.zeros((len(qlist), query_maxlength))
    for i in range(len(qlist)):
        for j in range(len(qlist[i])):
            if qlist[i][j] in word2id:
                qlist_index[i][j]=word2id[qlist[i][j]]
            else:
                qlist_index[i][j] = word2id['UNK']

    slist_index=np.zeros((len(slist),passage_maxlength))
    for i in range(len(slist)):
        slist_index[i][slist[i]]=1

    elist_index=np.zeros((len(elist),passage_maxlength))
    for i in range(len(elist)):
        elist_index[i][elist[i]]=1

    return [plist_index,qlist_index,slist_index,elist_index]

def SimplySplitData(data,num):
    train_data=[]
    test_data=[]
    for i in range(len(data)):
        train_data.append(data[i][:num])
        test_data.append(data[i][num:])
    return train_data,test_data

def Eva(res,data):
    count=0
    for i in range(len(res)):
        if res[i][0] == data[0][i] and res[i][1] == data[1][i]:
            count+=1
    return count,count/float(len(res))

def print_valid_res(self,res,filename):
    if len(res) == 2:
        res=np.transpose(res)
    with open(filename,'w') as fout:
        for r in res:
            fout.write(str(r[0])+'\t'+str(r[1])+'\n')



