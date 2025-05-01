import clshq_tk
from AUTODCETS import util, feature_selection, datasets, save_database as sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer

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

class train_Dataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, idx):
        #torch.cat((self.input_ids[idx], self.labels[idx]),axis=0)
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx].clone()
        }

    def __len__(self):
        return len(self.input_ids)

class test_Dataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        #torch.cat((self.input_ids[idx], self.labels[idx]),axis=0)
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
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


def create_sequences(X, y, tokenizer):
  sequences = []
  for i in range(len(X)):
      seq_in = X.iloc[i].values
      seq_out = y.iloc[i]
      sequences.append((str(seq_in).replace("'", ""),str(seq_out)))
  
  text_sequences = [f"{inp} {out}" + tokenizer.eos_token for inp, out in sequences]

  return text_sequences


def fuzzy_causal_tokenizer(df, name_dataset, target, max_lags, test_window_start, tokenizer, partitions):
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
    y_test = y[test_window_start:]

    # Sequence generation
    sequences = create_sequences(X, y_hat, tokenizer)
    sequences_train = sequences[:test_window_start]

    sequences_test = []
    for x in sequences[test_window_start:]:
        sequences_test.append(x.split("]")[0] + "]")

    # Tokenization
    tokenizer.pad_token = tokenizer.eos_token
    train_input_tokens = tokenizer(sequences_train, padding_side = 'left', padding=True, return_tensors="pt")
    test_input_tokens = tokenizer(sequences_test, padding_side = 'left', padding=True, return_tensors="pt")
    tokenized = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt") 
    
    return train_Dataset(train_input_tokens.input_ids, train_input_tokens.attention_mask), test_Dataset(test_input_tokens.input_ids, test_input_tokens.attention_mask, y_test), dict_variables

def train_model(train_dataset, name_model, epochs, path_model = None):
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(name_model, torch_dtype="auto",device_map="auto")

    # Training
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=50,
        weight_decay=0.001,
        logging_steps=500,
        prediction_loss_only=True,
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

def predict(train_dataset, model, tokenizer, dict_variables, target):

  dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

  all_preds = []
  all_actuals = []

  for batch_data in dataloader:
      inputs = batch_data['input_ids']
      attention_mask = batch_data['attention_mask']
      labels = batch_data['labels']

      inputs = inputs.to(DEVICE)
      attention_mask = attention_mask.to(DEVICE)

      outputs = model.generate(inputs, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id)

      decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

      preds = []
      for x in decoded:
        var, fset = dict_variables[target][2].get_fuzzy_set_by_name(x.split("]")[-1].strip())
        preds.append(dict_variables[target][2].centers[var][fset].item())
      
      l = [float(x) for x in labels]

      all_preds.extend(preds)
      all_actuals.extend(l)

    
  return all_preds, all_actuals

def rolling_window(df):
    import pandas as pd

    total_len = len(df)
    window_size = int(0.3 * total_len)
    n_windows = 10

    # Calcular passo necessário para obter 10 janelas
    step = (total_len - window_size) // (n_windows - 1)

    # Gerar janelas
    windows = [df.iloc[i:i + window_size] for i in range(0, step * (n_windows - 1) + 1, step)]

    return windows

def calc_metrics(database_path):
    import statistics
    
    datasets = pd.DataFrame(sd.execute("SELECT name_dataset FROM results", database_path), columns=['name_dataset'])['name_dataset'].unique().tolist()
    windows = pd.DataFrame(sd.execute("SELECT window FROM results", database_path), columns=['window'])['window'].unique().tolist()
    
    results_datasets = []
    for d in datasets:
        mae = []
        rmse = []
        for w in windows:
            query = "SELECT * FROM results WHERE name_dataset=='"+d+"' and window=="+str(w)
            results = pd.DataFrame(sd.execute(query, database_path), columns=['name_dataset', 'window', 'forecasts', 'real'])

            mae.append(np.mean(np.abs(np.array(results['forecasts'].values) - np.array(results['real'].values))))
            rmse.append(np.sqrt(np.mean((np.array(results['forecasts'].values) - np.array(results['real'].values)) ** 2)))

        avg_mae = statistics.mean(mae)
        avg_rmse = statistics.mean(rmse)

        std_mae = statistics.stdev(mae)
        std_rmse = statistics.stdev(rmse)

        df_resultados = pd.DataFrame([{
            "Dataset": d,
            "AVG RMSE": avg_rmse,
            "STD RMSE": std_rmse,
            "AVG MAE": avg_mae,
            "STD MAE": std_mae,
        }])

        results_datasets.append(df_resultados)

    return results_datasets
