import clshq_tk
from AUTODCETS import util, feature_selection, datasets, save_database as sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
import torch
from torch.utils.data import Dataset, DataLoader
from clshq_tk.modules.fuzzy import GridPartitioner, trimf, gaussmf, training_loop
from clshq_tk.data.regression import RegressionTS
from clshq_tk.common import DEVICE, DEFAULT_PATH, resume, checkpoint, order_window
import sys
import time
import signal
import gc

sys.path.append('Fuzzy-Causal-LLM')
import fuzzy_causal_text_lora as fcllm

"""
**Datasets**:
**1. CLIMATIC_1:** SONDA --- TARGET: glo_avg --- 35.000 SAMPLES --- 12 VARIABLES
**2. ENERGY_1**: WIND ENERGY PRODUCTION --- TARGET: Power --- 43.800 SAMPLES --- 9 VARIABLES
**3. IOT_1:** HOUSEHOLD ELECTRICITY CONSUMPTION IN MEXICO --- TARGET: active_power --- 100.000 SAMPLES --- 14 VARIABLES
**4. ECONOMICS_1:** BITCOIN --- TARGET: AVG --- 2.970 SAMPLES --- 06 VARIABLES
"""
def get_dataset(name_dataset):
    if name_dataset == 'SONDA':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/SONDA.csv', index_col='Date')
    elif name_dataset == 'WEC':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/WEC.csv', index_col='Date')
    elif name_dataset == 'DEC':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/DEC.csv', index_col='Date')
    elif name_dataset == 'AQ':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/AQ.csv', index_col='Date')
    elif name_dataset == 'WTH':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/WTH.csv', index_col='date')
    elif name_dataset == 'ETT':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/ETT.csv', index_col='date')
    else:
        return "There is no dataset with that name."
        
# Configuration
name_dataset = 'AQ'
target = 'PM2.5'
max_lags = 20  
partitions = 30
path_model = 'model'
database_path = 'database_text.db'
epochs = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name_model = "distilbert-base-uncased"  

# Print system info for debugging
if torch.cuda.is_available():
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA memory available: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

# Load dataset
df = get_dataset(name_dataset)
if isinstance(df, str):
    raise ValueError(f"Dataset loading failed: {df}")
# Load dataset
#df = fcllm.upload_dataset(name_dataset)

# Trim dataset based on type
# Trim dataset based on type
if name_dataset == 'CLIMATIC_1':
    df = df.head(35000)
elif name_dataset == 'ENERGY_1':
    df = df.head(43800)
elif name_dataset == 'IOT_1':
    df = df.head(100000)
elif name_dataset == 'AQ':
    df = df.head(35052)
else:
    df = df.head(2970)

print(f"Dataset: {name_dataset}, Shape: {df.shape}, Target: {target}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(name_model)

# Create rolling windows
windows = fcllm.rolling_window(df, 10)

# Initialize database and check for existing results
sd.execute("CREATE TABLE IF NOT EXISTS results(name_dataset TEXT, window INT, forecasts FLOAT, \
               real FLOAT)", database_path)

# Check which windows have already been processed
existing_windows = set()
try:
    existing_results = pd.DataFrame(sd.execute("SELECT DISTINCT window FROM results WHERE name_dataset=?", 
                                              database_path, (name_dataset,)), columns=['window'])
    existing_windows = set(existing_results['window'].tolist())
    print(f"Found existing results for windows: {sorted(existing_windows)}")
except:
    print("No existing results found. Starting from scratch.")

print("Starting experiment with rolling windows...")

# Experiment configuration options
USE_LORA = True  
EXPERIMENT_TYPE = "fuzzy_causal"

# Create custom LoRA configuration for DistilBERT
if USE_LORA:
    custom_lora_config = fcllm.create_lora_config(
        r=64,  
        lora_alpha=128,  
        lora_dropout=0.1, 
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"]  # DistilBERT-specific modules
    )
else:
    custom_lora_config = None

print(f"Configuration: LoRA={USE_LORA}, Experiment Type={EXPERIMENT_TYPE}")

# Loop through the windows with improved error handling
def timeout_handler(signum, frame):
    raise TimeoutError("Window processing timed out")

# Set timeout for each window (30 minutes)
for i, window in enumerate(windows):
    if i in existing_windows:
        print(f"Skipping window {i+1}/{len(windows)} (already processed)")
        continue

    print(f"\nProcessing window {i+1}/{len(windows)}...")

    try:
        # Set timeout for this window
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1800)  # 30 minutes timeout

       
        if EXPERIMENT_TYPE == "fuzzy_causal":
            ds, scaler = fcllm.fuzzy_causal(window, name_dataset, target, max_lags, tokenizer, partitions)
        elif EXPERIMENT_TYPE == "causal_text":
            ds, scaler = fcllm.causal_text(window, name_dataset, target, max_lags, tokenizer)
        else:  # Default to "text"
            ds, scaler = fcllm.text(window, name_dataset, target, max_lags, tokenizer)
        
        print(f"Dataset created with {len(ds)} samples")

        # Split dataset into train and test
        train_size = int(0.8 * len(ds))
        test_size = len(ds) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, test_size])
        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

        # Train model
        print("Starting model training...")
        model = fcllm.train_model(
            train_dataset,
            name_model,
            epochs,
            scaler,
            path_model=None,
            use_lora=USE_LORA,
            lora_config=custom_lora_config
        )

        # Make predictions
        print("Making predictions...")
        forecasts, real = fcllm.predict(test_dataset, model, tokenizer, target, scaler)

        # Save results to database
        for j in range(len(forecasts)):
            sd.execute_insert(
                "INSERT INTO results VALUES(?, ?, ?, ?)",
                (name_dataset, i, float(forecasts[j]), float(real[j])),
                database_path
            )

        print(f"Saved results for window {i}: {len(forecasts)} predictions")

        # Clean up memory
        del model, ds, train_dataset, test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Disable timeout
        signal.alarm(0)

    except TimeoutError:
        print(f"Timeout processing window {i}. Skipping...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue
    except Exception as e:
        print(f"Error processing window {i}: {str(e)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue
    finally:
        signal.alarm(0)  # Ensure timeout is disabled

print("\nExperiment completed!")

# Display results
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

# Show sample of results
results_df = pd.DataFrame(sd.execute("SELECT * FROM results", database_path), 
                         columns=['name_dataset', 'window', 'forecasts', 'real'])
print(f"\nTotal predictions saved: {len(results_df)}")
print("\nSample results:")
print(results_df.head(10))

# Calculate and display metrics
try:
    results_metrics = fcllm.calc_metrics(database_path)
    print("\nMetrics per dataset:")
    for result in results_metrics:
        print(result)
        print("-------")
except Exception as e:
    print(f"Error calculating metrics: {str(e)}")
    
print(f"\nResults saved to database: {database_path}")
print("Experiment configuration:")
print(f"- Dataset: {name_dataset}")
print(f"- Target: {target}")
print(f"- Max lags: {max_lags}")
print(f"- Epochs: {epochs}")
print(f"- LoRA enabled: {USE_LORA}")
print(f"- Experiment type: {EXPERIMENT_TYPE}")
