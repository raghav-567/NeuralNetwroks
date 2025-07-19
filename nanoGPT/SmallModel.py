import torch
import torch.nn as nn
from torch.nn import functional as F
max_iterations = 5000
batch_size = 64
learning_rate = 3e-4
block_size = 64
n_embd = 126
eval_iters = 200
eval_interval = 500
dropout = 0.2
no_of_head = 6

torch.manual_seed(1337)
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[i] for i in s]
decode = lambda l : ''.join((itos[i]) for i in l)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]

def get_split(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_split(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embd,head_size,bias=None)
        self.key = nn.Linear(n_embd,head_size,bias=None)
        self.value = nn.Linear(n_embd,head_size,bias=None)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = q @ k.transpose(-1,-2) * (C ** -0.5) 
        wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf'))
        wei = F.softmax(wei,dim = -1)
        wei = self.dropout(wei)
        out = wei @ v
        return out
    

class feedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.ffwd(x) 



class multihead_attention(nn.Module):
    def __init__(self, head_size , no_of_head):
        super().__init__()
        self.heads = nn.ModuleList(head(head_size) for _ in range(no_of_head))
        self.projection = nn.Linear(n_embd,n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = self.projection(torch.cat([h(x) for h in self.heads],dim = -1))
        out = self.dropout(out)
        return out



class block(nn.Module):
    def __init__(self,no_of_head,n_embd):
        super().__init__()
        head_size = n_embd//no_of_head
        self.sa = multihead_attention(head_size,no_of_head)
        self.ffwd = feedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embdedding = nn.Embedding(vocab_size,n_embd) 
        self.position_embd = nn.Embedding(block_size,n_embd) # for position of each char in block
        # self.sa_heads = multihead_attention(n_embd//4,4)
        # self.feedforward = feedForward(n_embd)
        self.block = nn.Sequential(
            block(no_of_head,n_embd),
            block(no_of_head,n_embd),
            block(no_of_head,n_embd),
            block(no_of_head,n_embd),
            block(no_of_head,n_embd),
            nn.LayerNorm(n_embd),
        )

        self.lmhead = nn.Linear(n_embd,vocab_size)

    def forward(self,idx,target=None):
        B,T = idx.shape  #C  -> n_embd
        pos_embd = self.position_embd(torch.arange(T, device=device))
 #T -> block size
        token_embd  = self.embdedding(idx) #(B,T,n_embd)
        x = token_embd + pos_embd #(B,T,C) 
        x = self.block(x)
        # x = self.sa_heads(x)
        # x = self.feedforward(x)
        logits = self.lmhead(x) #(B,T,C) -> (B,T,voacb_size)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = TransformerLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
# print(model.parameters)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iterations):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iterations - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_split('train')
    xb,yb = xb.to(device), yb.to(device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

