import pandas as pd
import numpy as np

from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from clshq_tk.modules.fuzzy import GridPartitioner, trimf, gaussmf, training_loop
from clshq_tk.data.regression import RegressionTS
from clshq_tk.common import DEVICE, DEFAULT_PATH, resume, checkpoint, order_window
from typing import List, Dict, Any
from AUTODCETS import feature_selection, datasets, util
import torch
from torch.utils.data import Dataset
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class TS_Dataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.input_ids)


def fuzzification(df, name_dataset, letter, partitioner = None):
    '''
        Fuzzification of the univariate times series.
        param df: DataFrame with the time series data.
        param name_dataset: Name of the dataset.
        param letter: Name of the linguistic variable.
        param partitioner: GridPartitioner object. If None, it will be created.
        return: Fuzzified time series, original time series labels, partitioner object.
    '''

    if partitioner is None:
        partitions = 25  # Number of fuzzy sets
        order = 1

        ts = RegressionTS(name_dataset, 100, df.values, order = order, step_ahead = 0, dtype=torch.float64)

        partitioner = GridPartitioner(trimf, partitions, ts.num_attributes, device = DEVICE, dtype = ts.dtype,
                                    var_names = [letter])
    else:
        ts = ts
        partitioner = partitioner

    training_loop(partitioner, ts)

    out = partitioner.forward(ts.y.to(device=DEVICE), mode = 'one-hot')

    ts_fuzzy = np.array(partitioner.from_membership_to_linguistic(out)).squeeze()

    return pd.DataFrame(ts_fuzzy), ts.y, partitioner


def create_sequences(X, y):
  sequences = []
  for i in range(len(X)):
      seq_in = X.iloc[i].values
      seq_out = y.iloc[i]
      sequences.append((str(seq_in).replace("'", ""),str(seq_out)))
  
  text_sequences = [f"{inp} {out}" for inp, out in sequences]

  return text_sequences


def fuzzy_causal_tokenizer(df, name_dataset, target, max_lags):
    variables = df.columns.tolist()
    dict_variables = dict.fromkeys(variables)
    
    # Fuzzification of time series
    data_fuzzy = pd.DataFrame(columns=variables)
    for v in variables:
        dict_variables[v] = fuzzification(pd.DataFrame(df[v]), name_dataset, v, None)
        data_fuzzy[v] = dict_variables[v][0]

    # Causal graph generation
    graph = feature_selection.causal_graph(df, target=target, max_lags=max_lags)[target]
    X, y_hat = util.organize_dataset(data_fuzzy, graph, max_lags, target)
    y = dict_variables[target][1].squeeze().tolist()[max_lags:]

    # Sequence generation
    sequences = create_sequences(X, y_hat)

    # Tokenization
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    return tokenized, tokenizer, y, dict_variables[target][2], sequences


def get_datasets(tokenized, y, test_window_start):

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    labels = input_ids.clone()
    labels[:, :1] = -100  # mask inputs

    train_dataset = TS_Dataset(input_ids[:test_window_start], attention_mask[:test_window_start], labels[:test_window_start])
    val_dataset = TS_Dataset(input_ids[test_window_start:], attention_mask[test_window_start:], labels[test_window_start:])
    y_val = y[test_window_start:]

    return train_dataset, val_dataset, y_val

def train_model(train_dataset, val_dataset, path_model):
    
    # Model
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)

    # Training
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=0.001,
        logging_steps=20,
        prediction_loss_only=True,
        report_to="none",
        save_strategy='no',
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    model.save_pretrained(path_model)

    return model

def predict(model, tokenizer, sequences, partitioner):
    pred_length = 1

    input_seq = sequences[:sequences.index(']') + 1]
        
    inputs = tokenizer.encode(input_seq, return_tensors='pt').to(DEVICE)

    outputs = model.generate(
        inputs,
        max_length = inputs.shape[1] + pred_length * 4,
        temperature=0.8,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    decoded_split = decoded.split(']')[-1].split()
        
    decoded_value = []
    for i in tuple(decoded_split):
        try:
            var, fset = partitioner.get_fuzzy_set_by_name(i)
            decoded_value.append(partitioner.centers[var][fset].item())
        except:
            print("Except")
            decoded_value.append(0)
            
    pred_out = np.mean(decoded_value)

    return pred_out