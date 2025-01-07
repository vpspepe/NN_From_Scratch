# GPT FROM SCRATCH: CHAR LEVEL GPT-2
# Lib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt.bigram import BigramModel
from gpt.funcs import train_test_val_split, get_batch, estimate_loss
import datetime as dt
import argparse
torch.manual_seed(42)

# Parse arguments
parser = argparse.ArgumentParser(description="Reuse or train new model")

# Add arguments
parser.add_argument("--model_file", type=str, help="Model file to reuse", required=False)
parser.add_argument("--text_file", type=str, help="Input text file to reuse", 
                    required=False, default="input.txt")
parser.add_argument("--hyperparams_file", type=str, help="Json file with hyperparams", 
                    required=False, default="bigram_params.json")

# Parse arguments
args = parser.parse_args()

# Input file: all Shakespeare books
text = open(args.text_file, 'r', encoding='utf-8').read()

# Vocabulary and encode/decode
chars = sorted(list(set(text)))
vocab_size = len(chars)

#Create a dictionary mapping each character to a unique integer
stoi = {ch: i for i,ch in enumerate(chars)}
itos = {v:k for k,v in stoi.items()}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If we are reusing a model
if args.model_file:
    # Load the model
    model = torch.load(args.model_file)
    model = model.to(device)
    
else:
    # Hyperparams
    hyperparams = json.loads(open(args.hyperparams_file, 'r').read())

    # Train Test Val Split
    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data, test_data = train_test_val_split(data, 
                                                        0.8, 
                                                        0.2, 
                                                        0)

    # Model
    model = BigramModel(vocab_size=vocab_size,
                        block_size=hyperparams["block_size"],
                        n_embd=hyperparams["n_embd"],
                        n_heads=hyperparams["n_head"],
                        n_layers=hyperparams["n_layer"],
                        dropout=hyperparams["dropout"]
                        )
    model = model.to(device)

    # Optimizing it
    optim = torch.optim.AdamW(model.parameters(), lr=hyperparams["learning_rate"])

    for iter in range(hyperparams["max_iters"]):

        # Sometimes do eval
        if iter % hyperparams["eval_interval"] == 0:
            losses = estimate_loss(model, hyperparams["eval_iters"], train_data, val_data, hyperparams["block_size"], hyperparams["batch_size"], device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Take batch
        xb, yb = get_batch('train', train_data, val_data, 
                        hyperparams["block_size"], hyperparams["batch_size"], device)

        # Measure the loss
        logits, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    model_name = f"bigram_model_"
    model_name += "_".join([f"{k}_{v}" for k,v in hyperparams.items()])
    model_name += f"_{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pth"
    torch.save(model, model_name)

# Sample from the model now
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))





