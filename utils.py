import pandas as pd
import numpy as np
import sqlite3
import contextlib
import torch

def get_dataset(name):
    
    if name == 'SONDA':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/SONDA.csv', index_col=('Date'))
    elif name == 'WEC':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/WEC.csv', index_col=('Date'))
    elif name == 'DEC':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/DEC.csv', index_col=('Date'))
    elif name == 'AQ':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/AQ.csv', index_col=('Date'))
    elif name == 'WTH':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/WTH.csv', index_col=('date'))
    elif name == 'ETT':
        return pd.read_csv('https://raw.githubusercontent.com/FutureLab-DCC/CGF-LLM/refs/heads/main/datasets/ETT.csv', index_col=('date'))
    else:
        return "There is no dataset with that name."


def execute_insert(sql,data,database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro data: dados que serão inseridos no banco de dados
    :parametro database_path: caminho para o banco de dados
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: # auto-closes
        with conn: # auto-commits
            with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                cursor.execute(sql,data)
                return cursor.fetchall()


def execute(sql, database_path):
    """
    Função para executar INSERT INTO
    :parametro sql: string com código sql
    :parametro database_path: caminho para o banco de dados
    :return: dataframe com os valores retornados pela consulta sql
    """
    with contextlib.closing(sqlite3.connect(database_path)) as conn: # auto-closes
        with conn: # auto-commits
            with contextlib.closing(conn.cursor()) as cursor: # auto-closes
                cursor.execute(sql)
                return cursor.fetchall()

def get_metrics(database_path):
    import statistics

    model = pd.DataFrame(execute("SELECT model FROM results", database_path), columns=['model'])['model'].unique().tolist()
    datasets = pd.DataFrame(execute("SELECT name_dataset FROM results", database_path), columns=['name_dataset'])['name_dataset'].unique().tolist()
    windows = pd.DataFrame(execute("SELECT window FROM results", database_path), columns=['window'])['window'].unique().tolist()

    results_datasets = []
    for m in model:
        for d in datasets:
            mae = []
            rmse = []
            nrmse = []
            for w in windows:
              try:
                  query = "SELECT * FROM results WHERE model=='"+m+"' and name_dataset=='"+d+"' and window=="+str(w)
                  results = pd.DataFrame(execute(query, database_path), columns=['model', 'name_dataset', 'window', 'forecasts', 'real'])
                  
                  rmse.append(np.sqrt(np.mean((np.array(results['forecasts'].values) - np.array(results['real'].values)) ** 2)))
                  maxmin = max(results['real'].values) - min(results['real'].values)
                  nrmse.append(np.sqrt(np.mean((np.array(results['forecasts'].values) - np.array(results['real'].values)) ** 2))/maxmin)
              except:
                  pass

            avg_nrmse = statistics.mean(nrmse)
            std_nrmse = statistics.stdev(nrmse)
    
            df_resultados = pd.DataFrame([{
                "Model": m,
                "Dataset": d,
                "AVG NRMSE": avg_nrmse,
                "STD NRMSE": std_nrmse,
            }])
    
            results_datasets.append(df_resultados)

    return results_datasets

def rolling_window(df, n_windows):
    import pandas as pd

    total_len = len(df)
    window_size = int(0.3 * total_len)

    # Calcular passo necessário para obter 10 janelas
    step = (total_len - window_size) // (n_windows - 1)

    # Gerar janelas
    windows = [df.iloc[i:i + window_size] for i in range(0, step * (n_windows - 1) + 1, step)]

    return windows

def get_metrics_multivariate(database_path):
    import statistics
    
    model = pd.DataFrame(execute("SELECT model FROM results", database_path), columns=['model'])['model'].unique().tolist()
    datasets = pd.DataFrame(execute("SELECT name_dataset FROM results", database_path), columns=['name_dataset'])['name_dataset'].unique().tolist()
    windows = pd.DataFrame(execute("SELECT window FROM results", database_path), columns=['window'])['window'].unique().tolist()
    
    nrmse_rows = []
    for m in model:
        for d in datasets:
    
            for w in windows:
            # try:
              query = "SELECT * FROM results WHERE model=='"+m+"' and name_dataset=='"+d+"' and window=="+str(w)
              results = pd.DataFrame(execute(query, database_path), columns=['model', 'name_dataset', 'window', 'variable', 'forecasts', 'real'])
    
              variables = results['variable'].unique()
    
              for v in variables:
                results_variable = results[results['variable'] == v]
                rmse = np.sqrt(np.mean((np.array(results_variable['forecasts'].values) - np.array(results_variable['real'].values)) ** 2))
                maxmin = max(results_variable['real'].values) - min(results_variable['real'].values)
                nrmse = np.sqrt(np.mean((np.array(results_variable['forecasts'].values) - np.array(results_variable['real'].values)) ** 2))/maxmin
                nrmse_rows.append([m, d, w, v, nrmse])
    nrmse_df_all = pd.DataFrame(nrmse_rows, columns=['model', 'name_dataset', 'window', 'variable', 'nrmse'])

    avg_rmse = nrmse_df_all.groupby(['model', 'name_dataset', 'variable'])['nrmse'].mean().reset_index()

    return avg_rmse

def get_sizes(database_size_path):
    query = "SELECT * FROM results"
    db = pd.DataFrame(execute(query, database_size_path), columns=['model', 'name_dataset', 'window', 'total_chars', 'train_token_count', 'test_token_count', 'train_char_count', 'test_char_count'])
    cols = ['total_chars', 'train_token_count', 'test_token_count', 'train_char_count', 'test_char_count']
    avg_sizes = db.groupby(['model', 'name_dataset'])[cols].mean().reset_index()

    return avg_sizes

def calculate_train_test_sizes(ds, seq, tokenizer):
    # Token size (total and average, with and without padding)
    total_tokens = sum(len(ds[i]['input_ids']) for i in range(len(ds)))
    total_non_padding_tokens = sum(
        len([t for t in ds[i]['input_ids'] if t != tokenizer.pad_token_id])
        for i in range(len(ds))
    )
    avg_tokens = total_tokens / len(ds)
    avg_non_padding_tokens = total_non_padding_tokens / len(ds)

    # Text size (total and average character length)
    total_chars = sum(len(text) for text in seq)
    avg_chars = total_chars / len(seq)


    # Split the dataset into train (80%) and test (20%)
    train_dataset, test_dataset = torch.utils.data.random_split(ds, [0.80, 0.20])

    # Token counts (including and excluding padding)
    train_token_count = sum(len(ds[i]['input_ids']) for i in train_dataset.indices)
    train_non_padding_token_count = sum(
        len([t for t in ds[i]['input_ids'] if t != tokenizer.pad_token_id])
        for i in train_dataset.indices
    )
    test_token_count = sum(len(ds[i]['input_ids']) for i in test_dataset.indices)
    test_non_padding_token_count = sum(
        len([t for t in ds[i]['input_ids'] if t != tokenizer.pad_token_id])
        for i in test_dataset.indices
    )

    # Text size (character length)
    train_texts, test_texts = torch.utils.data.random_split(seq, [0.80, 0.20])
    train_char_count = sum(len(text) for text in train_texts)
    test_char_count = sum(len(text) for text in test_texts)

    # Text size (total and average character length), Train tokens (including padding), Test tokens (including padding), Train text size (characters), Test text size (characters)
    return total_chars, train_token_count, test_token_count, train_char_count, test_char_count
