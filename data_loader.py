import csv
from collections import Counter
import random

import torch
import torch.nn.functional as F

def get_data(path):
        data=[]
        gold=[]
        with open(path) as fd:
                rd=csv.reader(fd, delimiter="\t", quotechar='"')
                for line in rd:
                        li=line[-1].split()
                        data.append(li)
                        gold.append(int(line[1]))
                        
        return data,gold

def get_vocab(data,min_count):
        word_to_int,int_to_word={},{}
        word_count=Counter()
        
        for sentence in data:
                for word in sentence:
                        word_count[word]+=1
                        
        res=[]
        for word in word_count:
                if word_count[word]>min_count:
                        res.append(word)
                        
        word_to_int['@pad'],word_to_int['@cls'],word_to_int['@unk']=0,1,2
        int_to_word[0],int_to_word[1],int_to_word[2]='@pad','@cls','@unk'
        
        index=3
        for word in res:
                int_to_word[index]=word
                word_to_int[word]=index
                index+=1
                
        return word_to_int,int_to_word
    
def accuracy_cal(output,answer):
        pred=F.softmax(output,dim=-1)
        _,pred=pred.max(dim=-1)
        count=0
        
        if pred[0]==answer[0]:
                count+=1
                        
        return count


class DataLoader(object):
        def __init__(self,data,gold,batch_size,
                     word_to_int,use_transformer):
                self.data=data
                self.batch_size=batch_size
                self.word_to_int=word_to_int
                self.gold=gold

                self.use_transformer=use_transformer
                
        def get_data(self):
                data=random.sample \
                (list(zip(self.data,self.gold)),self.batch_size)
                
                return data
            
        def __load_next__(self):
                data=self.get_data()
                
                dd_temp,data_temp,ans_temp,max_len=[],[],[],0
                for sentence in data:
                        temp=[]
                        sent=sentence[0]
                        
                        if self.use_transformer:
                                dd_temp.append(['@cls']+sent)
                                temp.append(self.word_to_int['@cls'])
                        else:
                                dd_temp.append(sent)

                        answer=sentence[1]
                        for word in sent:                               
                                if word in self.word_to_int:  
                                        temp.append(self.word_to_int[word])
                                else:
                                        temp.append(self.word_to_int['@unk'])
                        data_temp.append(temp)
                        ans_temp.append(answer)
                        max_len=max(max_len,len(temp))
                       
                input=torch.zeros(self.batch_size,max_len).long().cuda()
                input_mask=torch.zeros(self.batch_size,max_len).long().cuda()
                pos=torch.zeros(self.batch_size,max_len).long().cuda()
                answers=torch.tensor(ans_temp).long().cuda()
                
                for i,sentence in enumerate(data_temp):
                        input_mask[i][:len(sentence)]=1
                        
                        for j,word in enumerate(sentence):
                                input[i][j]=word
                                pos[i][j]=j
                                
                return dd_temp,input,input_mask,pos,answers
            
            
class TestLoader(DataLoader):
        def __init__(self,data,gold,word_to_int,use_transformer):
                self.data=data
                self.batch_size=1
                self.word_to_int=word_to_int
                self.gold=gold

                self.use_transformer=use_transformer
                self.counter=0
                self.len=len(data)
                
        def reset_counter(self):
                self.counter=0
                
        def get_data(self):
                data=self.data[self.counter]
                ans=self.gold[self.counter]   
                final=[(data,ans)]
                
                self.counter+=1
                if self.counter==len(self.data):
                        self.reset_counter()
                        
                return final