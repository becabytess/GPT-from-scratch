import torch

with open("tiny_shakspeare.txt",'r') as f:
          text = f.read()         
          
class Tokenizer:
    def __init__(self,vocab_size=26):
        self.vocab_size = 26 
        self.vocab = list(set(list(text)))
        self.vocab = sorted(self.vocab)
        self.vocab_size = len(self.vocab)
        self.token_to_id = {ch: i for i,ch in enumerate(self.vocab)}
        self.id_to_token = {i: ch for i,ch in enumerate(self.vocab)}
    def tokenize(self,text,return_tensor=True):
        ids = [self.token_to_id[ch] for ch in text]
        return torch.tensor(ids) if return_tensor else ids 




tokenizer = Tokenizer()
tokenized = tokenizer.tokenize(text)
CONTEXT_WINDOW  = 100 

def get_samples():
    batches = []
    for i in range(0,len(tokenized) - CONTEXT_WINDOW,CONTEXT_WINDOW):
        X = tokenized[i:i + CONTEXT_WINDOW]
        y = tokenized[i + 1:i + CONTEXT_WINDOW + 1]
        batches.append((X,y))
    batches = torch.stack([torch.stack(batch) for batch in batches])
    return batches 


class Dataset(torch.utils.data.Dataset):
    def __init__(self,batches):
        self.batches = batches
    def __len__(self):
        return len(self.batches)
    def __getitem__(self,idx):
        X,y = self.batches[idx]
        return X,y

all_batches = get_samples()
train_end = int(len(all_batches)*0.7)
train_data = Dataset(all_batches[:train_end])
train = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)

val_data = Dataset(all_batches[train_end:])
val  = torch.utils.data.DataLoader(val_data,batch_size=64,shuffle=True)

d_model=128

class Head:
    def __init__(self,d_model,dk):
        self.d_model = d_model
        self.q_w = torch.randn(d_model,dk)
        self.k_w = torch.randn(d_model,dk)
        self.v_w = torch.randn(d_model,dk)
    def forward(self,embedded_x):
        q = embedded_x @ self.q_w 
        k = embedded_x @ self.k_w 
        v = embedded_x @ self.v_w 

        attn_weights = q @ k.transpose(1,2) / (d_model ** 0.5)
        masked = attn_weights.tril()
        masked[masked == 0 ] = float('-inf')
        attn_scores = torch.nn.functional.softmax(masked,dim=-1)
        values = attn_scores @ v 
        return values 
    def params(self):
        return [self.q_w.flatten(),self.k_w.flatten(),self.v_w.flatten()]
    def __call__(self,Xs):
        return self.forward(Xs)
        
    
class MultiHeadAttention:
    def __init__(self,n_heads=8,d_model=128):
        
        
        
        self.n_heads = n_heads 
        self.dk = d_model // n_heads
        self.heads = [Head(d_model=d_model,dk=self.dk) for _ in range(n_heads)]
    def forward(self,h):
        values = torch.cat([head.forward(h) for head in self.heads],dim=-1)
        return values 
    def params(self):
        params = []
        for head in self.heads:
            params += head.params()
        return params
    def __call__(self,h):
        return self.forward(h)  + h


class FeedForward:
    def __init__(self,d_model=128,d_ff=512,n_layers=10):
        self.d_model = d_model 
        self.d_ff = d_ff 
        self.n_layers = n_layers
        weights = []
        biases = []
        
        weights.append(torch.randn(d_model,d_ff))
        biases.append(torch.randn(d_ff))

        for _ in range(self.n_layers):
            weights.append(torch.randn(d_ff,d_ff))
            biases.append(torch.randn(d_ff))


        self.proj_w = torch.randn(d_ff,d_model)
        self.proj_b = torch.randn(d_model)

        self.weights = weights 
        self.biases = biases
    def params(self):
        params = [self.proj_w.flatten(),self.proj_b.flatten()]
        for w,b in zip(self.weights,self.biases):
            params.append(w.flatten())
            params.append(b.flatten())
        return params 

    def forward(self,attn_out):
        for w,b in zip(self.weights,self.biases):
            attn_out = attn_out @ w + b 
            attn_out = torch.nn.functional.relu(attn_out)
        
        ff_out = attn_out @ self.proj_w + self.proj_b
        return ff_out 
    def __call__(self,attn_out):
        return self.forward(attn_out)

       
class TransformerBlock: 
    def __init__(self,d_model=128,d_ff=512,n_heads=8):
        self.attention = MultiHeadAttention(n_heads=n_heads,d_model=d_model)
        self.ff = FeedForward(d_model=d_model,d_ff=d_ff)
        self.attn_layer_norm = torch.nn.LayerNorm(d_model)
        self.ff_layer_norm = torch.nn.LayerNorm(d_model)
    def forward(self,h):
        attn_out = self.attention(h)
        attn_out = self.attn_layer_norm(attn_out)
        ff_out = self.ff(attn_out)
        ff_out = self.ff_layer_norm(ff_out)
        return ff_out 
    def params(self):
        params = []
        params += self.attention.params()
        params += self.ff.params()
        params += self.attn_layer_norm.parameters()
        params += self.ff_layer_norm.parameters()
        return params
    def __call__(self,h):
        return self.forward(h)



class GPT:
    def __init__(self,d_model=128,d_ff=512,n_heads=8,n_blocks=10):
        self.embedding = torch.randn(tokenizer.vocab_size, d_model)
        
        self.transformer_blocks = [TransformerBlock(d_model=d_model,d_ff=d_ff,n_heads=n_heads) for _ in range(n_blocks)]
        self.proj_w = torch.randn(d_model,tokenizer.vocab_size)
        self.proj_b = torch.randn(tokenizer.vocab_size)
        self.layer_norm = torch.nn.LayerNorm(tokenizer.vocab_size)
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    def forward(self,Xs):
        one_hot_x = torch.nn.functional.one_hot(Xs.long(),num_classes=tokenizer.vocab_size).float()
        embedded_x = one_hot_x @ self.embedding 
        for block in self.transformer_blocks:
            embedded_x = block(embedded_x)
        logits = embedded_x @ self.proj_w + self.proj_b 
        logits = self.layer_norm(logits)
        return logits 
    def __call__(self,Xs):
        return self.forward(Xs)
    
    def parameters(self):
        params = []
        params.append(self.proj_w.flatten())
        params.append(self.proj_b.flatten())
        params += self.layer_norm.parameters()
        for block in self.transformer_blocks:
            params += block.params()
        return params
    def train_step(self,Xs,ys):
        self.optimizer.zero_grad()
        logits = self.forward(Xs)
        loss = self.loss_fn(logits.view(-1,logits.shape[-1]),ys.flatten())
        loss.backward()
        self.optimizer.step()
        return loss.item()
    def evaluate(self):
        with torch.no_grad():
            val_loss = 0 
            for step, batch in enumerate(val):
                Xs,ys = batch
                logits = self.forward(Xs)
                loss = self.loss_fn(logits.view(-1,logits.shape[-1]),ys.flatten())
                val_loss += loss.item()
            val_loss /= 100
            print(f"Validation Loss: {val_loss}")
            return val_loss
    def train(self,epochs=100,log_steps=10,val_steps = 100,save_steps=100):
        for epoch in range(epochs):
            for step,batch in enumerate(train):
                
                Xs,ys = batch
                loss = self.train_step(Xs,ys)
                if (step + 1) % log_steps != 0:
                    print(f"Epoch {epoch + 1}, Batch {step + 1}, Loss: {loss}")
            
                if (step + 1) % val_steps ==0:
                    self.evaluate()
                if (step + 1) % save_steps ==0:
                    torch.save(self.state_dict(), f"gpt_epoch_{epoch + 1}_batch_{step + 1}.pt")
                    print(f"Model saved at epoch {epoch + 1}, batch {step + 1}")

    def state_dict(self):
        state = {}
        for i,param in enumerate(self.parameters()):
            state[f'param_{i}'] = param 
        return state
    def load_state_dict(self,state):
        for i,param in enumerate(self.parameters()):
            param.copy(state[f'param_{i}'])
        

   
model = GPT(d_model=128,d_ff=512,n_heads=8,n_blocks=10)
model.train(epochs=1,log_steps=10,val_steps=100,save_steps=100)
