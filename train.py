import torch.optim as optim
import torch.nn as nn

import argparse
import os

from data_loader import get_data,get_vocab,accuracy_cal
from data_loader import DataLoader,TestLoader

from BiLSTM import Model_B
from Transformer import Model_T

def main(args):
        data_train,gold_train=get_data(args.train_path)
        data_dev,gold_dev=get_data(args.dev_path)
        
        word_to_int,int_to_word=get_vocab(data_train,args.min_word_count)
        
        vocab_size=len(word_to_int)
        max_len=100
        
        train_loader=DataLoader(data_train,gold_train,args.batch_size,
                                word_to_int,args.transformer)
        dev_loader=TestLoader(data_dev,gold_dev,word_to_int,args.transformer)
        
        lossFunction=nn.CrossEntropyLoss()
        if args.transformer:
                model=Model_T(args.embed_size,args.hidden_size,args.inter_size,vocab_size,
                          max_len,args.n_heads,args.n_layers,args.per_layer,
                            args.dropout_prob_classifier,args.dropout_prob_attn,
                            args.dropout_prob_hidden,args.use_elmo,args.num_rep,args.elmo_drop).cuda()
        elif args.BiLSTM:
                model=Model_B(args.embed_size,args.hidden_size,vocab_size,
                            args.use_elmo,args.num_rep,args.elmo_drop).cuda()
        optimizer=optim.Adam(model.parameters(),lr=args.lr)
        
        train(model,optimizer,lossFunction,train_loader,dev_loader,args.epochs,args.eval_every)
        
def train(model,optimizer,lossFunction,train_loader,dev_loader,epochs,eval_every):
        for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()

                data,input_data,input_mask, \
                                positional,answers=train_loader.__load_next__()

                if args.transformer:
                        if args.use_elmo:  
                                output=model(input_data,positional,input_mask,data)
                        else:
                                output=model(input_data,positional,input_mask)
                elif args.BiLSTM:
                        if args.use_elmo:
                                output=model(input_data,input_mask,data)
                        else:
                                output=model(input_data,input_mask)

                loss=lossFunction(output,answers)
                scalar=loss.item()
                loss.backward()
                optimizer.step()
                
                print('epoch=',epoch+1,'training loss=',scalar)
                if (epoch+1)%eval_every==0:
                        validate(model,lossFunction,dev_loader)
                            
def validate(model,lossFunction,dev_loader):
        model.eval()
        scalar=0
        tn,tp,fn,fp=0,0,0,0

        for _ in range(dev_loader.len):
                    data,input_data,input_mask, \
                    positional,answers=dev_loader.__load_next__()

                    if args.transformer:
                        if args.use_elmo:  
                                output=model(input_data,positional,input_mask,data)
                        else:
                                output=model(input_data,positional,input_mask)
                    elif args.BiLSTM:
                        if args.use_elmo:
                                output=model(input_data,input_mask,data)
                        else:
                                output=model(input_data,input_mask)

                    loss=lossFunction(output,answers)
                    scalar+=loss.item()
                    acc=accuracy_cal(output,answers)

                    if answers[0]==1 and acc==1:
                            tp+=1
                    elif answers[0]==1 and acc==0:
                            fn+=1
                    elif answers[0]==0 and acc==1:
                            tn+=1
                    elif answers[0]==0 and acc==0:
                            fp+=1

        mcc=tp*tn-fp*fn
        den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        if den==0:
                den=1
        mcc=mcc/(den**0.5)

        print('validation loss=',scalar/dev_loader.len,
              'validation Mathews correlation coefficient=',mcc*100)

def setup():
        parser=argparse.ArgumentParser('Argument Parser')
        
        parser.add_argument('--batch_size',type=int,default=128)
        parser.add_argument('--lr',type=float,default=0.00005)
        parser.add_argument('--hidden_size',type=int,default=1024)
        parser.add_argument('--embed_size',type=int,default=128)
        parser.add_argument('--n_heads',type=int,default=8)
        parser.add_argument('--n_layer',type=int,defaults=8)
        parser.add_argument('--per_layer',type=int,default=1)
        parser.add_argument('--inter_size',type=int,default=512)
        parser.add_argument('--train_path',type=str,default=os.getcwd()+'/in_domain_train.tsv')
        parser.add_argument('--dev_path',type=str,default=os.getcwd()+'/in_domain_dev.tsv')
        parser.add_argument('--epochs',type=int,default=10000)
        parser.add_argument('--min_word_count',type=int,default=0)
        parser.add_argument('--eval_every',type=int,default=50)
        parser.add_argument('--dropout_prob_classifier',type=float,default=0.1)
        parser.add_argument('--dropout_prob_attn',type=float,default=0)
        parser.add_argument('--dropout_prob_hidden',type=float,default=0)
        parser.add_argument('--num_rep',type=int,default=1)
        parser.add_argument('--elmo_drop',type=float,default=0)
        parser.add_argument('--use_elmo',type=bool,default=True)
        parser.add_argument('--BiLSTM',type=bool,default=True)
        parser.add_argument('--transformer',type=bool,default=False)
        
        args=parser.parse_args()
        
        return args

if __name__=='__main__':
        args=setup()
        main(args)           