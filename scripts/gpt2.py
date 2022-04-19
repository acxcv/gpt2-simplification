import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F


# # open dataset
# with open('final_frame_300.pkl', 'rb') as infile:
#     df = pickle.load(infile)

# # create test set and subtract from df
# test_set = df.sample(n = 15634)
# df = df.loc[~df.index.isin(test_set.index)]

# # reset indexes
# test_set = test_set.reset_index()
# df = df.reset_index()

# # save training frame
# with open('train_frame.pkl', 'wb') as train_out:
#     pickle.dump(df, train_out) 

# # save test frame
# with open('test_frame.pkl', 'wb') as test_out:
#     pickle.dump(test_set, test_out) 

# load train frame
with open('train_frame.pkl', 'rb') as train_in:
    df = pickle.load(train_in)

# load test frame
with open('test_frame.pkl', 'rb') as test_in:
    test_set = pickle.load(test_in)


# define special tokens for tokenizer
special_tokens_dict = {'additional_special_tokens': ['<|startnormal|>', '<|startsimple|>']}

class SimplificationDataset(Dataset):
    
    def __init__(self, gpt2_type="gpt2", max_length=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.text_data = []

        for row in df[['normal', 'simple']].iterrows():
          self.text_data.append(torch.tensor(
                self.tokenizer.encode(f"<|startnormal|> {row[1][0]} <|startsimple|> {row[1][1]}", max_length=max_length, truncation=True)))

        self.sequences_count = len(self.text_data)
        

    def __len__(self):
        return self.sequences_count

    def __getitem__(self, item):
        return self.text_data[item]


print('create dataset')

# dataset = SimplificationDataset(gpt2_type="gpt2")

with open('SimplificationDataset.pkl', 'rb') as dataset_in:
    dataset = pickle.load(dataset_in)

# with open('SimplificationDataset.pkl', 'wb') as outfile:
#     pickle.dump(dataset, outfile)


# load tokenizer and add special tokens
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# load model and resize embeddings
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))


# use pack tensor method to speed up training
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None



def train(
    dataset, model, tokenizer,
    batch_size=16, epochs=5, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir="./model_saves/", output_prefix="thu_eve",
    test_mode=False,epoch_save=False,
):

    # set GPU as device for training
    device=torch.device("cuda")
    model = model.cuda()
    model.train()

    # adam optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)


    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None

        if epoch_save:
            import os
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


print('train model')

model = train(dataset, model, tokenizer, epoch_save=True)

model_name = 'simplification_gpt2_10_test_split'

torch.save(model.state_dict(), f"{model_name}-statedict.pt")

print(f'saving model as {model_name}')

