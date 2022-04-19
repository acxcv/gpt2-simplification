from cgi import test
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm, trange


# define special tokens
special_tokens_dict = {'additional_special_tokens': ['<|startnormal|>', '<|startsimple|>']}


# load and adapt model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load('simplification_gpt2_10_test_split-statedict.pt'))




def generate(model, tokenizer, prompt, entry_count=10, top_p=0.8, temperature=0.75,):

    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        for entry_idx in trange(entry_count):

            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(len(prompt)):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:

                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break

            if not entry_finished:
              output_list = list(generated.squeeze().numpy())
              output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
              generated_list.append(output_text)

    return generated_list


def generate_text(test_data='test_frame.pkl'):

    with open(test_data, 'rb') as test_in:
        test_set = pickle.load(test_in)
        
        generated_strings = []
        count = 0

        for row in test_set.iterrows():
            text_input = row[1]['normal']
            model_input = tokenizer.encode(f"<|startnormal|> {text_input} <|startsimple|> ", return_tensors='pt')
            model_output = model.generate(model_input, max_length=600, do_sample=True, top_k=40, temperature=0.7, no_repeat_ngram_size=2)
            text_output = tokenizer.decode(model_output[0], skip_special_tokens=True)

            generated_strings.append(text_output)

            count += 1
            if (count % 100) == 0:
                with open('generated_results.pkl', 'wb') as results_out:
                    pickle.dump(generated_strings, results_out)
        return generated_strings


# generate_text()
