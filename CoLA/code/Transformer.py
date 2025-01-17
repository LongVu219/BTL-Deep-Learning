import torch
import torch.nn as nn
import torch.nn.functional as F


class Embeddings(nn.Module):
        def __init__(self,vocab_size,max_len,
                     h_size,h_attn_size,
                     use_elmo,num_rep,elmo_drop):
                super(Embeddings,self).__init__()
                self.use_elmo=use_elmo
                
                self.token_embeds=nn.Embedding(vocab_size,h_size,padding_idx=0)
                self.pos_embeds=nn.Embedding(max_len,h_size+use_elmo*1024)
                self.layer_norm=nn.LayerNorm(use_elmo*1024+h_size)
                
                if use_elmo:
                        print('no u dont')

                self.project=nn.Linear(use_elmo*1024+h_size,h_attn_size)
                
        def forward(self,input,pos,data=None):
                if self.use_elmo:
                        print('no u dont')
                else:
                        rep=self.token_embeds(input)
                pos=self.pos_embeds(pos)
                
                output=self.layer_norm(rep+pos)
                output=self.project(output)
                
                return output
            
            
class SelfAttention(nn.Module):
        def __init__(self,h_size,n_heads,prob_attn,prob_h):
                super(SelfAttention,self).__init__()
                self.n_heads=n_heads
                self.h_size=h_size
                
                self.query=nn.Linear(h_size,h_size)
                self.key=nn.Linear(h_size,h_size)
                self.value=nn.Linear(h_size,h_size)

                self.dropout_attn=nn.Dropout(p=prob_attn)
                self.dropout_h=nn.Dropout(p=prob_h)
                self.out=nn.Linear(h_size,h_size)
                
                self.layer_norm=nn.LayerNorm(h_size)
                
        def forward(self,input,input_mask):
                qq=self.query(input)
                kk=self.key(input)
                vv=self.value(input)
                
                qq=qq.view(input.shape[0],-1,self.n_heads,self.h_size//self.n_heads)
                kk=kk.view(input.shape[0],-1,self.n_heads,self.h_size//self.n_heads)
                vv=vv.view(input.shape[0],-1,self.n_heads,self.h_size//self.n_heads)
                
                qq=qq.transpose(1,2)
                kk=kk.transpose(1,2)
                vv=vv.transpose(1,2)
                
                interact=torch.matmul(qq,kk.transpose(-1,-2))
                attn_weights=F.softmax(interact,dim=-1)
                mask_1=input_mask.unsqueeze(-1).unsqueeze(1).expand(-1,self.n_heads,-1,input.shape[1])
                mask_2=input_mask.unsqueeze(1).unsqueeze(1).expand(-1,self.n_heads,input.shape[1],-1)
                attn_weights=attn_weights*(mask_1*mask_2)
                attn_weights=self.dropout_attn(attn_weights)
                
                output=torch.matmul(attn_weights,vv)
                output=output.transpose(1,2)
                output=output.contiguous().view(input.shape[0],-1,self.h_size)
                
                output=self.dropout_h(self.out(output))
                output=self.layer_norm(output+input)
                
                return output
            
            
class Intermediate(nn.Module):
        def __init__(self,inter_size,h_size):
                super(Intermediate,self).__init__()
                
                self.linear=nn.Linear(h_size,inter_size)
                self.act=nn.GELU()
                
        def forward(self,input):
                output=self.linear(input)
                output=self.act(output)
                
                return output
            
            
class FFN(nn.Module):
        def __init__(self,h_size,inter_size):
                super(FFN,self).__init__()
                
                self.linear=nn.Linear(inter_size,h_size)
                self.layernorm=nn.LayerNorm(h_size)
                
        def forward(self,input,attn_output):
                output=self.linear(input)
                output=self.layernorm(output+attn_output)
                
                return output
            
            
class Layer(nn.Module):
        def __init__(self,h_size,inter_size,
                     n_heads,prob_attn,prob_h):
                super(Layer,self).__init__()
                
                self.attn=SelfAttention(h_size,n_heads,prob_attn,prob_h)
                self.inter=Intermediate(inter_size,h_size)
                self.ffn=FFN(h_size,inter_size)
                
        def forward(self,input,input_mask):
                attn=self.attn(input,input_mask)
                inter=self.inter(attn)
                output=self.ffn(inter,attn)
                
                return output
            

class Pooler(nn.Module):
        def __init__(self,h_size,prob,n_options=2):
                super(Pooler,self).__init__()
                
                self.project=nn.Linear(h_size,n_options)
                self.dropout=nn.Dropout(p=prob)
                
        def forward(self,input):
                output=input[:,0,:].view(input.shape[0],1,-1)
                output=self.dropout(output)
                output=self.project(output).squeeze(1)
                
                return output

           
class Model_T(nn.Module):
        def __init__(self,embed_size,h_size,inter_size,vocab_size,
                     max_len,n_heads,n_layers,per_layer,prob_cl,prob_attn,prob_h,
                     use_elmo,num_rep=None,elmo_drop=None):
                super(Model_T,self).__init__()
                self.embed=Embeddings(vocab_size,max_len,
                                      embed_size,h_size,
                                      use_elmo,num_rep,elmo_drop)
                
                self.layer=nn.ModuleList([Layer(h_size,inter_size,
                                                n_heads,prob_attn,prob_h) for _ in range(n_layers)])
                self.per_layer=per_layer
                
                self.pooler=Pooler(h_size,prob_cl,2)
                
        def forward(self,token,pos,input_mask,data=None):
                output=self.embed(token,pos,data)

                for layer in self.layer:
                        for _ in range(self.per_layer):
                                output=layer(output,input_mask)
         
                output=self.pooler(output)
                                
                return output