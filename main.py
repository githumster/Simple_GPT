import torch
from model import GPTLanguageModel
from utils import load_data, prepare_data, get_batch, estimate_loss

n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

text = load_data('all_tracks.txt')
train_data, val_data, vocab_size, encode, decode = prepare_data(text)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPTLanguageModel(vocab_size, n_embd, n_layer, n_head, block_size=256, dropout=dropout).to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model, train_data, val_data, eval_iters)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(context, max_new_tokens=500)
print(decode(generated_text[0].tolist()))
