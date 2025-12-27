import tiktoken 
import random 
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split 
import numpy as np 
import os 
from pathlib import Path


enc = tiktoken.get_encoding('cl100k_base')
torch.set_default_device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_source = os.path.join('datasets','tiny_shakspeare.txt')

train_data_path =os.path.join('datasets', 'train','train_gpt.bin')

test_data_path =os.path.join('datasets', 'val','test_gpt.bin')



train_ratio = 0.9

if not Path(train_data_path).is_file():
    with open(f'{data_source}','r') as f: 
        dataset = f.read()  
        encoded_dataset = enc.encode(dataset)
        train_end = int(train_ratio*len(encoded_dataset))
        train_data = encoded_dataset[:train_end]
        val_data = encoded_dataset[train_end:]
        np.array(train_data,dtype=np.uint32).tofile(train_data_path)
        np.array(val_data,dtype=np.uint32).tofile(test_data_path)





class TextDataset(torch.utils.data.Dataset):
    def __init__(self,path,seq_len):
        self.data = np.memmap(path,dtype=np.uint32,mode="r") 
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self,idx):
        x =torch.from_numpy(self.data[idx:idx+self.seq_len].copy()).long()
        y =torch.from_numpy(self.data[idx + 1: idx + 1 + self.seq_len].copy()).long()

        return x, y 








# text_dataset = TextDataset("tiny_shakespeare.txt")


#training params 

batch_size = 50 
lr  = 1e-3
seq_len = 200

train_dataset = TextDataset(train_data_path,seq_len)
test_dataset = TextDataset(test_data_path,seq_len)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)




 
# ids = enc.encode(text)


class Attention(nn.Module): 
    def __init__(self,head_size, embed_size,seq_len):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_size,head_size)
        self.key = nn.Linear(embed_size,head_size)
        self.value = nn.Linear(embed_size,head_size)
        self.embed_size = embed_size
        
        
        #
        
        
    def forward(self,embds):
        query = self.query(embds)
        key = self.key(embds)
        value = self.value(embds)
        scores = query @ key.transpose(-2,-1) / self.head_size ** 0.5
        # print(embds.shape)
        mask = torch.tril(torch.ones(embds.shape[1],embds.shape[1]))
        
        
        scores = scores.masked_fill(mask==0,float('-inf'))
        attention_weights = torch.softmax(scores,dim=-1)
        attention = attention_weights @ value 
        return attention 

        


class MultiHeadAttention(nn.Module): 
    def __init__(self,head_size,n_head, embed_size,seq_len):
        super().__init__()
        self.head_size = head_size 
        self.n_head = n_head 
        self.heads = nn.ModuleList([Attention(self.head_size, embed_size,seq_len) for _ in range(self.n_head)])
        
    

    def forward(self,embds):
        attn_out = torch.cat([h(embds) for h in self.heads],dim=-1)
        return attn_out 



class Block(nn.Module): 
    def __init__(self,embed_size,n_head,seq_len):
        super().__init__()
        self.embed_size = embed_size 
        self.n_head = n_head 
        self.head_size = embed_size // n_head 
        self.multi_head_attention = MultiHeadAttention(self.head_size,self.n_head, self.embed_size,seq_len)
        self.ff = nn.Sequential(
            nn.Linear(self.embed_size, 4*self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size*4,self.embed_size),
            nn.Dropout(0.2),
            ) 
        

        self.ln1 = nn.LayerNorm(self.embed_size)
        self.ln2 = nn.LayerNorm(self.embed_size)
    def forward(self,x):    
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x
       




 
class Transformer(nn.Module):
    def __init__(self,embed_size,n_head,n_block,vocab_size,seq_len):
        super().__init__()
        self.embed_size = embed_size 
        self.n_head = n_head 
        self.n_block = n_block
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        self.model = nn.Sequential (*[
          Block(self.embed_size,self.n_head,self.seq_len) for _ in range(self.n_block)
        ])
       
 

        self.embedding = nn.Embedding(vocab_size,embed_size)
        #learnt positional encoding like gpt2 and gpt 3
        self.positional_encoding = nn.Embedding(seq_len,embed_size)
        self.proj = nn.Linear(embed_size,vocab_size)
        

    def forward(self,ids):
        
        embedding = self.embedding(ids)
        positional_encoding = self.positional_encoding(torch.arange(ids.shape[-1]))
        embedding = embedding + positional_encoding
        outs = self.model(embedding)
        logits =self.proj(outs)
        return logits

    def generate(self,inputs,max_tokens):
        with torch.no_grad():

            
            
           
            text_tensor = inputs[0].unsqueeze(0)
            tokens = 0 
            i = 0 
            j= text_tensor.shape[-1]
            initial_tokens = j
            print(text_tensor.shape)
            while tokens < max_tokens: 

                logits = self.forward(text_tensor[:,i:j])
                probs = F.softmax(logits)
                last_token = probs[:,-1,:]
                best = [torch.multinomial(last_token,num_samples=1)]
                print(enc.decode(best),end='')
                best = torch.tensor(best)
                
                text_tensor = torch.cat((text_tensor,best.unsqueeze(0)),-1)
                tokens += 1
                j += 1

                if tokens  + initial_tokens >= seq_len:
                    i += 1



         

        



    def test(self):
        total_loss= 0
        num_batches = 0
        
        for x,y in test_loader:
               

                num_batches +=  1
                with torch.no_grad(): 
                    logits = self.forward(x)
                    y = y.flatten()
                    logits = logits.view(-1,logits.shape[-1])
                    
                    total_loss += F.cross_entropy(logits,y)

                    if num_batches > int(0.9*len(test_loader)):
                        self.generate(x)

        loss = total_loss/ num_batches
        return loss
        


    def fit(self,epochs=40):
        

        
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-2,weight_decay=1e-2)
        
        for epoch in range(epochs): 
            total_loss = 0 
            num_batches = 0 
            for x,y in train_loader:

                num_batches += 1
                logits = self.forward(x)
                y = y.flatten()
                logits = logits.view(-1,logits.shape[-1])
                
                loss = F.cross_entropy(logits,y)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if num_batches % 20 == 0 :
                    print(f"step {num_batches}/{len(train_loader)} , train loss: {loss}")

                if num_batches % 100 == 0 :
                    print('Testing...')
                    test_loss = self.test()
                    
                    print(f"step {num_batches}/{len(train_loader)} , train loss: {loss}    , Test loss: {test_loss}")

                    


           

        





model  = Transformer(128,4,4,enc.n_vocab,seq_len)

# model.fit()
# torch.save(model.state_dict(),'final.pt')



inputs = ['hello world is the first']
encoded_text = enc.encode_batch(inputs)
encoded_text = torch.tensor(encoded_text)
# print(encoded_text)
model.generate(encoded_text,100)


















        




