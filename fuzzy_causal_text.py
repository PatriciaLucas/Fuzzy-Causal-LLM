import clshq_tk
from AUTODCETS import util, feature_selection, datasets, save_database as sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import GPT2Model, GPT2Config

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from clshq_tk.modules.fuzzy import GridPartitioner, trimf, gaussmf, training_loop
from clshq_tk.data.regression import RegressionTS
from clshq_tk.common import DEVICE, DEFAULT_PATH, resume, checkpoint, order_window


def upload_dataset(name_dataset):
  return pd.DataFrame(datasets.get_multivariate(name_dataset))

def upload_dataset(name_dataset):
  return pd.DataFrame(datasets.get_multivariate(name_dataset))

class custom_Dataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        #torch.cat((self.input_ids[idx], self.labels[idx]),axis=0)
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.Tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.input_ids)

def fuzzification(df, name_dataset, letter, partitions, partitioner = None):

    if partitioner is None:

        ts = RegressionTS(name_dataset, 2000, df.values, order = 1, step_ahead = 0, dtype=torch.float64)

        partitioner = GridPartitioner(trimf, partitions, ts.num_attributes, device = DEVICE, dtype = ts.dtype,
                                    var_names = [letter])
    else:
        ts = ts
        partitioner = partitioner

    training_loop(partitioner, ts)

    out = partitioner.forward(ts.y.to(device=DEVICE), mode = 'one-hot')

    ts_fuzzy = np.array(partitioner.from_membership_to_linguistic(out)).squeeze()

    return pd.DataFrame(ts_fuzzy), ts.y, partitioner




def create_sequences_input(X, y, tokenizer):
  sequences = []
  for i in range(len(X)):
      seq_in = X.iloc[i].values
      seq_out = y[i]
      sequences.append((str(seq_in).replace("'", ""),str(seq_out)))
  
  text_sequences = [f"{inp} {out}" + tokenizer.eos_token for inp, out in sequences]

  return text_sequences


def causal_text(df, name_dataset, target, max_lags, test_window_start, tokenizer):
                
    variables = df.columns.tolist()
    dict_variables = dict.fromkeys(variables)
    
    # Fuzzification of time series
    # data_fuzzy = pd.DataFrame(columns=variables)
    # for v in variables:
    #     dict_variables[v] = fuzzification(pd.DataFrame(df[v]), name_dataset, v, partitions, None)
    #     data_fuzzy[v] = dict_variables[v][0]

    # Causal graph generation
    graph = feature_selection.causal_graph(df.head(2000), target=target, max_lags=max_lags)[target]
    X, y_hat = util.organize_dataset(df, graph, max_lags, target)
    y = df[target].squeeze().tolist()[max_lags:]
    y = np.asarray(y)
    # y_test = y[test_window_start:]

    scaler = StandardScaler()
    labels_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # Sequence generation
    # sequences = create_sequences_text(X, y_hat, tokenizer)
    # sequences_train = sequences[:test_window_start]

    # sequences_test = []
    # for x in sequences[test_window_start:]:
    #     sequences_test.append(x.split("]")[0] + "]")

    inputs = create_sequences_input(X, y, tokenizer)

    print(inputs[0])

    # Tokenization
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer(inputs, padding_side = 'left', padding=True, return_tensors="pt")
    
    return custom_Dataset(input_tokens.input_ids, input_tokens.attention_mask, labels_scaled), scaler

def text(df, name_dataset, target, max_lags, test_window_start, tokenizer):

    # Complete graph generation
    graph = feature_selection.complete_graph(df.head(2000), target=target, max_lags=max_lags)[target]
    X, y_hat = util.organize_dataset(df, graph, max_lags, target)
    y = df[target].squeeze().tolist()[max_lags:]
    y = np.asarray(y)

    scaler = StandardScaler()
    labels_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # # Sequence generation
    # sequences = create_sequences_text(X, y_hat, tokenizer)
    # sequences_train = sequences[:test_window_start]
    
    # sequences_test = []
    # for x in sequences[test_window_start:]:
    #     sequences_test.append(x.split("]")[0] + "]")
    # print(sequences_test[0])
    
    inputs = create_sequences_input(X, y, tokenizer)

    print(inputs[0])

    # Tokenization
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer(inputs, padding_side = 'left', padding=True, return_tensors="pt")
    
    return custom_Dataset(input_tokens.input_ids, input_tokens.attention_mask, labels_scaled), scaler


def fuzzy_causal(df, name_dataset, target, max_lags, test_window_start, tokenizer):
                
    variables = df.columns.tolist()
    dict_variables = dict.fromkeys(variables)
    
    # Fuzzification of time series
    data_fuzzy = pd.DataFrame(columns=variables)
    for v in variables:
        dict_variables[v] = fuzzification(pd.DataFrame(df[v]), name_dataset, v, partitions, None)
        data_fuzzy[v] = dict_variables[v][0]

    # Causal graph generation
    graph = feature_selection.causal_graph(df.head(2000), target=target, max_lags=max_lags)[target]
    X, y_hat = util.organize_dataset(data_fuzzy, graph, max_lags, target)
    y = dict_variables[target][1].squeeze().tolist()[max_lags:]
    y = np.asarray(y)

    scaler = StandardScaler()
    labels_scaled = scaler.fit_transform(y.reshape(-1, 1))

    # Sequence generation
    # sequences = create_sequences_fuzzy(X, y_hat, tokenizer)
    # sequences_train = sequences[:test_window_start]

    # sequences_test = []
    # for x in sequences[test_window_start:]:
    #     sequences_test.append(x.split("]")[0] + "]")

    inputs = create_sequences_input(X, y, tokenizer)

    # Tokenization
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer(inputs, padding_side = 'left', padding=True, return_tensors="pt")
    
    return custom_Dataset(input_tokens.input_ids, input_tokens.attention_mask, labels_scaled), scaler

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):
        attn_scores = self.attn(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (attn_weights * x).sum(dim=1)
        return pooled

class GPT2Forecaster(nn.Module):
    def __init__(self, scaler, output_size=1, hidden_dims=[128, 64]):
        super().__init__()
        config = GPT2Config()
        self.gpt2 = GPT2Model(config)
        self.scaler = scaler

        # Attention-based pooling over the hidden states
        self.attn_pool = AttentionPooling(config.n_embd)

        # MLP head for forecasting
        layers = []
        in_dim = config.n_embd
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, labels):

        gpt_outputs = self.gpt2(input_ids, attention_mask = attention_mask)
        hidden_states = gpt_outputs.last_hidden_state  # [batch_size, seq_len, embed_dim]
        

        pooled = self.attn_pool(hidden_states)  # [batch_size, embed_dim]
        
        output = self.mlp_head(pooled)  # [batch_size, output_size]
        

        
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()

            loss = loss_fn(output, labels)
        
        
        return {"loss" : loss, "logits" : output.squeeze(0)}

def train_model(train_dataset, name_model, epochs, path_model = None):
    
    # Model
    model = GPT2Forecaster(scaler=scaler)
    # model = AutoModelForCausalLM.from_pretrained(name_model, torch_dtype="auto",device_map="auto")

    # Training
    training_args = TrainingArguments(
        output_dir='/kaggle/working/',
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=0.001,
        logging_steps=20,
        report_to="none",
        save_strategy='no',
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        #eval_dataset=val_dataset,
    )

    trainer.train()

    if path_model is not None:
        model.save_pretrained(path_model)

    return model



def predict(test_dataset, model, tokenizer, target, dict_variables = None):

  dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

  all_preds = []
  all_actuals = []

  for batch_data in dataloader:
    inputs = batch_data['input_ids'].to(DEVICE)
    attention_mask = batch_data['attention_mask'].to(DEVICE)
    labels = batch_data['labels'].to(DEVICE)

    outputs = model.forward(inputs, attention_mask=attention_mask, labels=None)

    predictions = outputs['logits'].detach().cpu()
    true = labels.detach().cpu()
    unscaled = scaler.inverse_transform(predictions)
    trues = scaler.inverse_transform(true)

    all_preds.extend(unscaled.squeeze(1))
    all_actuals.extend(trues.squeeze(1))

    
  return all_preds, all_actuals
