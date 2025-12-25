import tiktoken 
import random 
import torch.nn as nn
import torch.nn.functional as F
import torch
with open(r'C:\Users\beca\Desktop\sfs\ebt\tiny_shakspeare.txt','r',encoding='utf-8') as f: 
    text = f.read()



enc = tiktoken.get_encoding('cl100k_base')
ids = enc.encode(text)


class Attention(nn.Module): 
    def __init__(self,head_size, embed_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_size,head_size)
        self.key = nn.Linear(embed_size,head_size)
        self.value = nn.Linear(embed_size,head_size)
        
    def forward(self,embds):
        query = self.query(embds)
        key = self.key(embds)
        value = self.value(embds)

        scores = query @ key.transpose(-2,-1) / self.head_size ** 0.5
        mask = torch.tril(torch.ones_like(scores))
        scores = scores.masked_fill(mask==0,float('-inf'))
        attention_weights = torch.softmax(scores,dim=-1)
        
        attention = attention_weights @ value 
        return attention 

        


class MultiHeadAttention(nn.Module): 
    def __init__(self,head_size,n_head, embed_size):
        super().__init__()
        self.head_size = head_size 
        self.n_head = n_head 
        self.heads = nn.ModuleList([Attention(self.head_size, embed_size) for _ in range(self.n_head)])
        
    

    def forward(self,embds):
        attn_out = torch.cat([h(embds) for h in self.heads],dim=-1)
        return attn_out 



class Block(nn.Module): 
    def __init__(self,embed_size,n_head):
        super().__init__()
        self.embed_size = embed_size 
        self.n_head = n_head 
        self.head_size = embed_size // n_head 
        self.multi_head_attention = MultiHeadAttention(self.head_size,self.n_head, self.embed_size)
        self.ff = nn.Sequential(
            nn.Linear(self.embed_size, 4*self.embed_size),
            nn.GELU(),
            nn.Linear(self.embed_size*4,self.embed_size)) 

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
          Block(self.embed_size,self.n_head) for _ in range(self.n_block)
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

    def generate(self,ids):
        probs = self.forward(ids)
        last_token = probs[-1]
        best = torch.argmax(last_token)
        return enc.decode([best.tolist()])

    def train(self,train_data: torch.Tensor,epochs=10,seq_len=1024,batch_size=200):

        xs = []
        ys = []
        
        optimizer = torch.optim.AdamW(self.parameters(),lr=1e-3,weight_decay=1e-2)
        for i in range(0,train_data.shape[-1],seq_len):
            x = train_data[i:i+seq_len]
            y = train_data[i+1:i+seq_len+1]
            if x.shape[-1]<seq_len: 
                break 
            xs.append(x)
            ys.append(y)

        xs = torch.stack(xs) 
        ys = torch.stack(ys)
        
        for epoch in range(epochs): 
            print(f'staring epoch : {epoch+1}')

            for i in range(0,len(xs),batch_size):
                x = xs[i: i + batch_size]
                y = ys[i: i + batch_size]

                logits = self.forward(x)
                loss = F.cross_entropy(logits,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if (epoch + 1) % 10 ==0 : 
                    print(f"Epoch : {epoch} , Loss: {loss}")






model  = Transformer(128,8,12,enc.n_vocab,1024)
ids = torch.tensor(ids)
# print(ids.shape)


model.train(ids)










        




