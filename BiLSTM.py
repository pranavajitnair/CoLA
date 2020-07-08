import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from allennlp.modules.elmo import batch_to_ids,Elmo


class LSTM(nn.Module):
        def __init__(self,input_size,hidden_size):
                super(LSTM,self).__init__()
                
                self.lstm=nn.LSTM(input_size,hidden_size,bidirectional=True,batch_first=True,num_layers=2)
                
        def forward(self,input,input_mask):
                seq_len=torch.sum(input_mask,dim=-1)
                sorted_len,sorted_index=seq_len.sort(0,descending=True)
                i_sorted_index=sorted_index.view(-1,1,1).expand_as(input)
                sorted_input=input.gather(0,i_sorted_index.long())
                
                packed_seq=pack_padded_sequence(sorted_input,sorted_len,batch_first=True)
                output,(hidden,cell_state)=self.lstm(packed_seq)
                unpacked_seq,unpacked_len=pad_packed_sequence(output,batch_first=True)
                
                _,original_index=sorted_index.sort(0,descending=False)
                unsorted_index=original_index.view(-1,1,1).expand_as(unpacked_seq)
                output_final=unpacked_seq.gather(0,unsorted_index.long())
                
                return output_final,seq_len
            

class Model_B(nn.Module):
        def __init__(self,embed_size,h_size,vocab_size,
                     use_elmo,num_rep=None,elmo_drop=None):
                super(Model_B,self).__init__()

                options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
                weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
                
                self.elmo_size=1024
                self.h_size=h_size
                self.use_elmo=use_elmo
                
                if self.use_elmo:
                        self.elmo=Elmo(options_file,weight_file,num_rep,dropout=elmo_drop)
                
                self.embeddings=nn.Embedding(vocab_size,embed_size,padding_idx=0)
                self.lstm=LSTM(embed_size+self.use_elmo*self.elmo_size,h_size)
                self.output=nn.Linear(2*h_size,2)
                
        def forward(self,input,input_mask,data=None):
                if self.use_elmo:
                        character_ids=batch_to_ids(data).cuda()
                        rep=self.elmo(character_ids)['elmo_representations'][0]
                        token_embed=self.embeddings(input)
                        final_embed=torch.cat([token_embed,rep],dim=-1)
                else:
                        final_embed=self.embeddings(input)
                
                output,_=self.lstm(final_embed,input_mask)
                
                k_mask=torch.sum(input_mask,dim=-1)
                arange=torch.arange(0,input.shape[1]*input.shape[0],input.shape[1]).cuda()
                k_mask=k_mask+arange-1
                output_final=output.view(-1,2*self.h_size).index_select(0,k_mask).view(-1,2*self.h_size)
                
                output_final=self.output(output_final)
                
                return output_final