import torch 
from tqdm import tqdm
class Tokenizer:
    def __init__(self,train_path=None,tokenizer_path=None):
        self.file = train_path 
        self.start = "<START>"
        self.end = "<END>"
        if train_path and tokenizer_path:
            raise ValueError("Please provide either train_path or tokenizer_path but not both.") 
        tokenizer = None
        if tokenizer_path is not None:
            tokenizer = torch.load(tokenizer_path)

            self.token_to_id = tokenizer
            self.vocab = list(self.token_to_id.keys())
        else:
        
            with open(train_path,"r") as f:
                self.text = f.read()

                self.vocab = list(set(list(self.text)))
                self.vocab.sort()

        self.vocab= [self.start] + self.vocab + [self.end]
        if not tokenizer:
            self.token_to_id = {ch:i for i ,ch in enumerate(self.vocab)}

        self.id_to_token = {v:k for k ,v in self.token_to_id.items()} 
    def train(self,N=100,save_file_name="tokenizer"):
        words = self.text.split()
        
        tokenized = []
        for word in words: 
            tokenized_word =[self.token_to_id[self.start]]  + [self.token_to_id[ch] for ch in word] + [self.token_to_id[self.end]]
            tokenized.append(tokenized_word)
        tokenized = [token for word in tokenized for token in word]

        
        
        for _ in range(N):
            print(f"Iteration {_+1}/{N}")
            pairs = [(tokenized[i],tokenized[i+1]) for i in range(len(tokenized)-1)]
            count = {}
            for pair in pairs:
                if pair == (self.token_to_id[self.end],self.token_to_id[self.start]):
                    continue
                count[pair] = count.get(pair,0) + 1
            count = sorted(count.items(),key=lambda x: x[1],reverse=True)
            top_pair = count[0][0]
            new_token = self.id_to_token[top_pair[0]] + self.id_to_token[top_pair[1]]
            self.vocab.append(new_token)
            self.token_to_id[new_token] = len(self.vocab) -1 
            self.id_to_token[len(self.vocab)-1] = new_token

            updated_tokenized = []
            i = 0 
            while i < len(tokenized):
        
                
                if i < len(tokenized) - 1 and (tokenized[i],tokenized[i + 1]) == top_pair:
                    updated_tokenized.append(self.token_to_id[new_token])
                    i += 2
                else:
                    updated_tokenized.append(tokenized[i]) 
                    i += 1
            tokenized = updated_tokenized
        torch.save(self.token_to_id,f"{save_file_name}.pt") 
        print(f"Tokenizer saved as {save_file_name}.pt")
            
    def encode(self,text):
        ids = []
        i = 0
        pbar = tqdm(total=len(text),desc="Encoding text")

        while i < len(text):
            end = len(text) - 1
            sub = None
                
            while sub not in self.vocab:
                    
                sub = text[i:end + 1]
                end -=1 
            ids.append(self.token_to_id[sub])
                
            i += len(sub) 
            pbar.update(len(sub))

        return ids 
    def decode(self,ids):
        return ''.join([self.id_to_token[id] for id in ids])

   