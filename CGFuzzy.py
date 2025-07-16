from AUTODCETS import util, feature_selection
import numpy as np
import pandas as pd
from pyFTS.partitioners import Grid
from torch.utils.data import Subset

def fuzzyfy(df, partitioner = 'Grid', npart = 30):

  # Fuzzification of time series
  fs = pd.DataFrame(columns=df.columns.to_list())
  ts_fuzzy = pd.DataFrame(columns=df.columns.to_list())
  for v in df.columns.to_list():
      if partitioner == 'Grid':
        fs[v] = [Grid.GridPartitioner(data = df[v].values, npart=npart)]


      fuzzyfied = fs[v][0].fuzzyfy(data=df[v].values, method='maximum', mode='sets')
      fuzzyfied = fs[v][0].fuzzyfy(data=df[v].values, method='maximum', mode='sets')
      ts_fuzzy[v] = fuzzyfied

  return ts_fuzzy, fs

def fit(df, ts_fuzzy, target, max_lags):
  graph = feature_selection.causal_graph(df.head(2000), target=target, max_lags=max_lags)[target]
  antecedent, consequent = util.organize_dataset(ts_fuzzy, graph, max_lags, target)

  return [antecedent, consequent], graph

def calculate_weights(df):
    contagem = df.value_counts().reset_index()
    contagem.columns = ['sets', 'weights']
    return contagem

def predict(df, model, graph, max_lags, target, partitioners):
    input =  util.organize_dataset(df, graph, max_lags, target)[0]
    forecasts = []
    m = False
    for row in range(input.shape[0]):

      forecast_fuzzy = []
      for v in range(len(input.columns)):
          min = partitioners[input.columns[v]][0].min
          max = partitioners[input.columns[v]][0].max
          forecast_fuzzy.append(partitioners[input.columns[v]][0].fuzzyfy(data=float(input.iloc[row, v]), method='maximum', mode='sets'))


      # print(forecast_fuzzy)
      if not isinstance(forecast_fuzzy, pd.Series):
            linha = pd.Series(forecast_fuzzy, index=input.columns)

      centers, partlen = np.linspace(partitioners[target][0].min, partitioners[target][0].max, partitioners[target][0].partitions, retstep=True)

      equal = model[0].eq(linha).all(axis=1)

      if not equal.any():
        linha = model[0].iloc[-1]
        equal = model[0].eq(linha).all(axis=1)
        m = False

      sets_predict = calculate_weights(model[1][equal])

      weights_list = []
      center_list = []
      for index, sets in sets_predict.iterrows():
          idx_center = int(sets['sets'][1:])
          weights = sets['weights']
          center_list.append(float(centers[idx_center]) * weights)
          weights_list.append(weights)


      forecast = float(np.sum(center_list) / np.sum(weights_list))

      forecasts.append(forecast)

    return forecasts


