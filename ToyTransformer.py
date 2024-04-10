import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset


from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file="C:\\Users\\PUGGERNAUT\\Desktop\\c4-bpe\\tokenizer.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("Is she fast", tokenizer.is_fast)


import time


import math
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, activation, dim_feedforward, max_seq_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model).cuda()
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_length, d_model)).cuda()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        ).cuda()
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers).cuda()
        self.output_layer = nn.Linear(d_model, vocab_size).cuda()  # Final output layer
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def generate_square_subsequent_mask(self, sz):
        # Generates a square mask for the sequence to mask out subsequent positions
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, src, tgt,  tgt_mask=None, tgt_key_padding_mask=None):
        assert src.max() < self.vocab_size, "src contains invalid token indices"
        assert tgt.max() < self.vocab_size, "tgt contains invalid token indices"
        assert src.min() >= 0, "src contains negative indices"
        assert tgt.min() >= 0, "tgt contains negative indices"
        #print("Max input token index:", src.max().item(), "Max output token index:", src.max().item(), "Max input sequence length:", src.size(1), "Max output sequence length:", src.size(1))
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        # Adding positional encoding
        src = src + self.pos_encoder[:, :src.size(1)]
        tgt = tgt + self.pos_encoder[:, :tgt.size(1)]
        #output = self.transformer(src=src, tgt=tgt)

         # Create masks if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Since there is no encoder, the memory is just a dummy tensor with proper dimensions
        memory = torch.zeros((1, tgt.size(1), self.d_model), device=tgt.device)
        
        # Pass through the transformer decoder
        output = self.transformer_decoder(tgt=src, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.output_layer(output)
        return output
model = TransformerDecoderModel(d_model=256, nhead=1, num_decoder_layers=1, dim_feedforward=512, activation=torch.nn.ReLU, vocab_size=100608, max_seq_length=2048).cuda()


total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameters:", total_params)


optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

from transformers import DataCollatorForLanguageModeling
dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


from tqdm import tqdm
from random import randint

def remove_negative_indices(tensor):
    """Replace negative indices in a tensor with a valid index, such as the padding index."""
    padding_index = 0  # Assuming 0 is your padding index; adjust as needed
    tensor[tensor < 0] = padding_index
    return tensor

log_interval = 200
max_seq_length = 512
model.train()  # turn on train mode
#model = torch.compile(model)
LOSS = 10

checkpoint = torch.load("modelB_latest.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for file_name_index in range(5, 255):
    validation_file = "E:\\c4\\en\\c4-train." + str("0" * 4) + str(randint(0,8)) + "-of-01024.json.gz"
    v_dataset = load_dataset('json', data_files=validation_file)
    v_dataset = v_dataset.remove_columns(["url", "timestamp"])
    start_time = time.time()
    v_dataset['train'] = v_dataset['train'].map(lambda e: tokenizer(e['text']), batched=True)
    end_time = time.time()
    print("Tokenization of validation data took", end_time-start_time, "seconds")
    v_dataset['train'] = v_dataset['train'].remove_columns(["text", "token_type_ids", "attention_mask"])
    val_dl = DataLoader(v_dataset['train'], shuffle=True, batch_size=12, collate_fn=dc)
    training_files = ["E:\\c4\\en\\c4-train." + str("0" * (5-len(str(x)))) + str(x) + "-of-01024.json.gz" for x in range(file_name_index*4, (file_name_index+1)*4)]
    t_dataset = load_dataset('json', data_files=training_files)
    t_dataset = t_dataset.remove_columns(["url", "timestamp"]) #prune dataset for efficiency especially during tokenization
    start_time = time.time()
    t_dataset['train'] = t_dataset['train'].map(lambda e: tokenizer(e['text']), batched=True)
    end_time = time.time()
    print("Tokenization of training data took", end_time-start_time, "seconds")
    t_dataset['train'] = t_dataset['train'].remove_columns(["text", "token_type_ids", "attention_mask"])
    dc = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dl = DataLoader(t_dataset['train'], shuffle=True, batch_size=12, collate_fn=dc)
    training_steps = len(train_dl)
    progress_bar = tqdm(range(training_steps))
    for index, batch in enumerate(train_dl):
        if (index % 10000 == 0):
            if index != 0:
              print("\nAnother 6.1 million tokens bite the dust")
            loss = 0
            model.eval()
            val_examples = 1000
            with torch.no_grad():
                for index, batch in enumerate(val_dl):
                  if index > val_examples:
                    break
                  batch["labels"] = remove_negative_indices(batch["labels"]).cuda()

                  outputs = model(src=batch["input_ids"][:, :max_seq_length].cuda(), tgt=batch["labels"][:, :max_seq_length].cuda())
                  loss_fn = nn.CrossEntropyLoss(ignore_index=50304)

                  # Flatten the output and target tensors
                  outputs_flattened = outputs.view(-1, outputs.shape[-1])  # Shape: [sequence_length * batch_size, vocab_size]
                  targets_flattened = batch["labels"][:, :max_seq_length].reshape(-1)  # Shape: [sequence_length * batch_size]
                  loss += loss_fn(outputs_flattened, targets_flattened)

            #  Compute the loss
            avg_loss = loss/val_examples
            print("Loss:", avg_loss)
            torch.save({
            'epoch': file_name_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            }, "modelB"+str(int(time.time()))+".pt")

            torch.save({
            'epoch': file_name_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            }, "modelB_latest"+ ".pt")
        # Run a batch through the model
        model.train()
        batch["labels"] = remove_negative_indices(batch["labels"]).cuda()

        outputs = model(src=batch["input_ids"][:, :max_seq_length].cuda(), tgt=batch["labels"][:, :max_seq_length].cuda())
        loss_fn = nn.CrossEntropyLoss(ignore_index=50304)

        # Flatten the output and target tensors
        outputs_flattened = outputs.view(-1, outputs.shape[-1])  # Shape: [sequence_length * batch_size, vocab_size]
        targets_flattened = batch["labels"][:, :max_seq_length].reshape(-1)  # Shape: [sequence_length * batch_size]

        #  Compute the loss
        loss = loss_fn(outputs_flattened, targets_flattened)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
