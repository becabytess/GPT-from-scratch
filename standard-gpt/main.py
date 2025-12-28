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
# torch.set_default_device("cuda")
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
seq_len =1000
gradient_accumulation_steps = 40
checkpoint_interval = 1000
# lr_scheduler_step_size = 10 
# lr_scheduler_gamma = 0.9

#attempt to implement warmup 
warmup_steps = 2000 
max_lr = 1e-1
min_lr = 1e-4





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
        self.head_size = head_size 

        
        
        
        
        
    def forward(self,embds):
        query = self.query(embds)
        key = self.key(embds)
        value = self.value(embds)
        scores = query @ key.transpose(-2,-1) / self.head_size ** 0.5
        
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
            nn.Dropout(0.2))

        

        self.ln1 = nn.LayerNorm(self.embed_size)
        self.ln2 = nn.LayerNorm(self.embed_size)  
    def forward(self,x):    
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x
       




 
class Transformer(nn.Module):
    def __init__(self,embed_size,n_head,n_block,vocab_size,seq_len):
        super().__init__()
        assert embed_size % n_head ==0 , "Embed size must be divisible by n_head"
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
        self.apply(self._init_weights)
        self.optimizer = torch.optim.AdamW(params=self.parameters(),lr=1e-2,weight_decay=1e-2)
        
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=lr_scheduler_step_size,gamma=lr_scheduler_gamma)


        
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



         

        

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())

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
        

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
                nn.init.normal_(module.weight,mean=0.0,std=0.02)
                if module.bias is not None: 
                    nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
                nn.init.normal_(module.weight,mean=0.0,std=0.02)
        elif isinstance(module,nn.LayerNorm):
                nn.init.zeros_(module.bias) 
                nn.init.zeros_(module.weight)
    
            




    def fit(self,epochs=40):
        

        total_steps = epochs * len(train_loader)
        # total_decay_steps = total_steps
        steps = 0 
        for epoch in range(epochs): 
            total_loss = 0 
            num_batches = 0 
            for x,y in train_loader:
                steps += 1
                num_batches += 1
                logits = self.forward(x)
                y = y.flatten()
                logits = logits.view(-1,logits.shape[-1])
                
                loss = F.cross_entropy(logits,y)
                total_loss += loss.item()
                loss.backward()

                if num_batches % gradient_accumulation_steps ==0: 
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # self.lr_scheduler.step() 
                    if steps  < warmup_steps:
                        self.optimizer.param_groups[0]['lr'] = max_lr * (steps / warmup_steps)
                    elif self.optimizer.param_groups[0]['lr'] < min_lr: 
                        self.optimizer.param_groups[0]['lr']  = min_lr 
                    else:
                        #apply linear lr  decay 
                        self.optimizer.param_groups[0]['lr'] = max_lr - (max_lr - min_lr) * (steps / total_decay_steps)
                        


                
                if num_batches % 20 == 0 :
                    print(f"step {num_batches}/{len(train_loader)} , train loss: {loss}")

                if num_batches % 100 == 0 :
                    print('Testing...')
                    test_loss = self.test()
                    
                    print(f"step {num_batches}/{len(train_loader)} , train loss: {loss}    , Test loss: {test_loss}")

                if num_batches %  checkpoint_interval == 0 :
                    torch.save(self.state_dict(),os.path.join('checkpoints',f'checkpoint_{num_batches}.pt'))
                    print(f"Checkpoint saved at step {num_batches}")


                

        





model  = Transformer(120,10,30,enc.n_vocab,seq_len)

# model.fit()
# torch.save(model.state_dict(),'final.pt')

n_params = model.get_params_count()
print(f"Number of parameters: {n_params}")



inputs = ['hello world is the first']
encoded_text = enc.encode_batch(inputs)
encoded_text = torch.tensor(encoded_text)
# print(encoded_text)
model.generate(encoded_text,100)


















        




