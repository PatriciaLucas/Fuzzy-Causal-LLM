# -*- coding: utf-8 -*-
import clshq_tk
from AUTODCETS import util, feature_selection, datasets, save_database as sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, DistilBertModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from clshq_tk.modules.fuzzy import GridPartitioner, trimf, gaussmf, training_loop
from clshq_tk.data.regression import RegressionTS
from clshq_tk.common import DEVICE, DEFAULT_PATH, resume, checkpoint, order_window
import os
import sys

# Verify RegressionTS import
try:
    from clshq_tk.data.regression import RegressionTS
except ImportError as e:
    print(f"Error importing RegressionTS: {e}")
    raise ImportError("Could not import RegressionTS from clshq_tk.data.regression. Ensure clshq_tk is installed correctly.")

def upload_dataset(name_dataset):
    return pd.DataFrame(datasets.get_multivariate(name_dataset))

class custom_Dataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.Tensor(self.labels[idx])
        }
        return item

    def __len__(self):
        return len(self.input_ids)

def fuzzification(df, name_dataset, letter, partitions, partitioner=None):
    try:
        if partitioner is None:
            ts = RegressionTS(name_dataset, 2000, df.values, order=1, step_ahead=0, dtype=torch.float64)
            partitioner = GridPartitioner(trimf, partitions, ts.num_attributes, device=DEVICE, dtype=ts.dtype,
                                         var_names=[letter])
        else:
            ts = None

        if ts is not None:
            training_loop(partitioner, ts)
            out = partitioner.forward(ts.y.to(device=DEVICE), mode='one-hot')
            ts_fuzzy = np.array(partitioner.from_membership_to_linguistic(out)).squeeze()
            return pd.DataFrame(ts_fuzzy), ts.y, partitioner
        else:
            raise ValueError("No valid time series data provided for fuzzification.")
    except Exception as e:
        print(f"Error in fuzzification for variable {letter}: {str(e)}")
        raise

def create_sequences_input(X, y, tokenizer):
    sequences = []
    for i in range(len(X)):
        seq_in = X.iloc[i].values
        seq_out = y[i]
        sequences.append((str(seq_in).replace("'", ""), str(seq_out)))
    
    text_sequences = [f"{inp} {out}" + (tokenizer.sep_token or '</s>') for inp, out in sequences]
    return text_sequences

def fuzzy_causal(df, name_dataset, target, max_lags, tokenizer, partitions):
    variables = df.columns.tolist()
    dict_variables = dict.fromkeys(variables)
    
    data_fuzzy = pd.DataFrame(columns=variables)
    for v in variables:
        dict_variables[v] = fuzzification(pd.DataFrame(df[v]), name_dataset, v, partitions, None)
        data_fuzzy[v] = dict_variables[v][0]

    graph = feature_selection.causal_graph(df.head(2000), target=target, max_lags=max_lags)[target]
    X, y_hat = util.organize_dataset(data_fuzzy, graph, max_lags, target)
    y = dict_variables[target][1].squeeze().tolist()[max_lags:]
    y = np.asarray(y)

    scaler = StandardScaler()
    labels_scaled = scaler.fit_transform(y.reshape(-1, 1))

    inputs = create_sequences_input(X, y, tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = '</s>'
    input_tokens = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=1024,  
        return_tensors="pt",
        add_special_tokens=True
    )
    
    input_lengths = [len(ids) for ids in input_tokens.input_ids]
    print(f"Sequence length stats: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths):.1f}")
    if max(input_lengths) > 1024:
        print(f"Warning: Maximum sequence length ({max(input_lengths)}) exceeds max_length=1024. Consider reducing partitions or max_lags.")

    return custom_Dataset(input_tokens.input_ids, input_tokens.attention_mask, labels_scaled), scaler

def causal_text(df, name_dataset, target, max_lags, tokenizer):
    variables = df.columns.tolist()
    dict_variables = dict.fromkeys(variables)

    graph = feature_selection.causal_graph(df.head(2000), target=target, max_lags=max_lags)[target]
    X, y_hat = util.organize_dataset(df, graph, max_lags, target)
    y = df[target].squeeze().tolist()[max_lags:]
    y = np.asarray(y)

    scaler = StandardScaler()
    labels_scaled = scaler.fit_transform(y.reshape(-1, 1))

    inputs = create_sequences_input(X, y, tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = '</s>'
    input_tokens = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=1024,  
        return_tensors="pt",
        add_special_tokens=True
    )
    
    input_lengths = [len(ids) for ids in input_tokens.input_ids]
    print(f"Sequence length stats: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths):.1f}")
    if max(input_lengths) > 1024:
        print(f"Warning: Maximum sequence length ({max(input_lengths)}) exceeds max_length=1024. Consider reducing partitions or max_lags.")

    return custom_Dataset(input_tokens.input_ids, input_tokens.attention_mask, labels_scaled), scaler

def text(df, name_dataset, target, max_lags, tokenizer):
    graph = feature_selection.complete_graph(df.head(2000), target=target, max_lags=max_lags)[target]
    X, y_hat = util.organize_dataset(df, graph, max_lags, target)
    y = df[target].squeeze().tolist()[max_lags:]
    y = np.asarray(y)

    scaler = StandardScaler()
    labels_scaled = scaler.fit_transform(y.reshape(-1, 1))
    
    inputs = create_sequences_input(X, y, tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = '</s>'
    input_tokens = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=1024,  
        return_tensors="pt",
        add_special_tokens=True
    )
    
    input_lengths = [len(ids) for ids in input_tokens.input_ids]
    print(f"Sequence length stats: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths):.1f}")
    if max(input_lengths) > 1024:
        print(f"Warning: Maximum sequence length ({max(input_lengths)}) exceeds max_length=1024. Consider reducing partitions or max_lags.")

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

class DistilBertForecaster(nn.Module):
    def __init__(self, name_model, scaler, output_size=1, hidden_dims=[128, 64], use_lora=True, lora_config=None):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(name_model)
        self.scaler = scaler
        self.use_lora = use_lora

        if use_lora:
            if lora_config is None:
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                    r=64,  # Preserved from original script
                    lora_alpha=128,  # Preserved from original script
                    lora_dropout=0.1,  # Preserved from original script
                    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"]  # DistilBERT-specific modules
                )
            
            self.distilbert = get_peft_model(self.distilbert, lora_config)
            print(f"LoRA applied. Trainable parameters: {self.distilbert.get_nb_trainable_parameters()}")

        self.attn_pool = AttentionPooling(self.distilbert.config.hidden_size)

        layers = []
        in_dim = self.distilbert.config.hidden_size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_size))
        self.mlp_head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, labels=None):
        distilbert_outputs = self.distilbert(input_ids, attention_mask=attention_mask)
        hidden_states = distilbert_outputs.last_hidden_state
        pooled = self.attn_pool(hidden_states)
        output = self.mlp_head(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, labels)
        
        return {"loss": loss, "logits": output}

def train_model(train_dataset, name_model, epochs, scaler, path_model=None, use_lora=True, lora_config=None):
    from transformers import TrainingArguments, Trainer
    import torch

    model = DistilBertForecaster(name_model=name_model, scaler=scaler, use_lora=use_lora, lora_config=lora_config)

    # Debug: Inspect model structure
    print(f"Model type: {type(model)}")
    print(f"DistilBert type: {type(model.distilbert)}")
    if use_lora:
        print(f"LoRA model type: {type(model.distilbert.base_model) if hasattr(model.distilbert, 'base_model') else 'No base_model'}")

    # Enable gradient checkpointing on the DistilBERT model
    if hasattr(model.distilbert, "gradient_checkpointing_enable"):
        model.distilbert.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on DistilBertModel.")
    else:
        print("Warning: Could not enable gradient checkpointing on DistilBertModel. Proceeding without it.")

    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,  
        per_device_eval_batch_size=32,  
        gradient_accumulation_steps=2, 
        learning_rate=2e-4 if use_lora else 2e-5, 
        weight_decay=0.001, 
        logging_steps=20,  
        report_to="none",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    print(f"Starting training with {len(train_dataset)} samples...")
    try:
        trainer.train()
        print(f"Training completed. Peak memory: {torch.cuda.max_memory_allocated(0) / 1024**3:.1f} GB")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

    if path_model is not None:
        try:
            if use_lora:
                model.distilbert.save_pretrained(path_model)
            else:
                model.save_pretrained(path_model)
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    return model

def predict(test_dataset, model, tokenizer, target, scaler, dict_variables=None):
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # Preserved from original
    all_preds = []
    all_actuals = []

    for batch_data in dataloader:
        inputs = batch_data['input_ids'].to(DEVICE)
        attention_mask = batch_data['attention_mask'].to(DEVICE)
        labels = batch_data['labels'].to(DEVICE)

        outputs = model.forward(inputs, attention_mask=attention_mask, labels=None)
        predictions = outputs['logits'].detach().cpu()
        true = labels.detach().cpu()
        unscaled = scaler.inverse_transform(predictions.squeeze(-1).numpy().reshape(-1, 1))
        trues = scaler.inverse_transform(true.squeeze(-1).numpy().reshape(-1, 1))

        all_preds.extend(unscaled.flatten())
        all_actuals.extend(trues.flatten())

    return all_preds, all_actuals

def rolling_window(df, n_windows):
    total_len = len(df)
    window_size = int(0.3 * total_len)
    step = (total_len - window_size) // (n_windows - 1)
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
        nrmse = []
        for w in windows:
            try:
                query = "SELECT * FROM results WHERE name_dataset=='" + d + "' and window==" + str(w)
                results = pd.DataFrame(sd.execute(query, database_path), columns=['name_dataset', 'window', 'forecasts', 'real'])
                rmse.append(np.sqrt(np.mean((np.array(results['forecasts'].values) - np.array(results['real'].values)) ** 2)))
                maxmin = max(results['real'].values) - min(results['real'].values)
                nrmse.append(np.sqrt(np.mean((np.array(results['forecasts'].values) - np.array(results['real'].values)) ** 2)) / maxmin)
            except:
                pass
        
        avg_nrmse = statistics.mean(nrmse)
        std_nrmse = statistics.stdev(nrmse)
        df_resultados = pd.DataFrame([{
            "Dataset": d,
            "AVG NRMSE": avg_nrmse,
            "STD NRMSE": std_nrmse,
        }])
        results_datasets.append(df_resultados)

    return results_datasets

def create_lora_config(r=64, lora_alpha=128, lora_dropout=0.1, target_modules=None):
    if target_modules is None:
        target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]  # DistilBERT-specific modules
    
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
    )
