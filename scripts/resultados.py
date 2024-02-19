import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


paleta_cores = ['#2a8b8b','#75c58e','#bfff91','#dfe9a8','#ffd2bf']
dataset = pd.read_csv("/home/balboni/scripts/dataset_normalizado.txt",sep=';')
#print(dataset)
salvar_path = '/home/balboni/resultados/graficos/'
colunas = dataset.columns
tabela_melhores = pd.DataFrame(columns = colunas)
del(tabela_melhores['pdb'])
del(tabela_melhores['valor real'])
for _ in range(10):
    tabela_melhores.loc[len(tabela_melhores)] = ''


tabela_piores = tabela_melhores.copy()

for column in tabela_melhores:
    data_aux = dataset[[column, 'valor real','pdb']]
    #print(data_aux)
    data_aux['diff'] = abs(data_aux['valor real'] - data_aux[column])
    data_aux = data_aux.sort_values(by='diff').reset_index()
    #print(data_aux)
    for j in range(10):
        tabela_melhores[column][j] = data_aux['pdb'][j]
        tabela_piores[column][j] = data_aux['pdb'][len(data_aux)-j-1]
    #break

#print(tabela_melhores)
fig, ax = plt.subplots(figsize=(10, 5)) 
ax.axis('off')
ax.table(cellText=tabela_melhores.values, colLabels=tabela_melhores.columns, cellLoc='center', loc='center')

plt.savefig(salvar_path+'Melhores predições funcoes de escore.png', format='png', bbox_inches='tight')
plt.clf()




#print(tabela_piores)
fig, ax = plt.subplots(figsize=(10, 5)) 
ax.axis('off')
ax.table(cellText=tabela_piores.values, colLabels=tabela_piores.columns, cellLoc='center', loc='center')

plt.savefig(salvar_path+'Piores predições funcoes de escore.png', format='png', bbox_inches='tight')
plt.clf()

try:
    del(dataset['pdb'])
except:
    pass




















funcoes = dataset.columns[:-1]
resultado_rmse = []
for i in range(len(funcoes)):
    #print(funcoes[i], mean_squared_error(dataset[funcoes[i]], dataset['valor real']))
    #print(dataset[funcoes[i]])
    #print(funcoes[i], np.sqrt(mean_squared_error(dataset[funcoes[i]], dataset['valor real'])))
    resultado_rmse.append( np.sqrt(mean_squared_error(dataset[funcoes[i]], dataset['valor real'])))

plt.bar(funcoes, resultado_rmse, color=paleta_cores)

plt.ylabel('RMSE')
for i in range(len(funcoes)):
    plt.text(i,  resultado_rmse[i]/2, np.round(resultado_rmse[i],3), ha = 'center', color='black')
plt.xlabel('Funções de escore')
plt.xticks(rotation=70)
plt.title('Gráfico de RMSE das funções de score')


plt.savefig(salvar_path+'RMSE.png', format='png', bbox_inches='tight')
plt.clf()

pearson = []
for i in range(len(funcoes)):
    #print(funcoes[i], mean_squared_error(dataset[funcoes[i]], dataset['valor real']))
    pearson.append(dataset[funcoes[i]].corr(dataset['valor real']))
plt.bar(funcoes, pearson, color=paleta_cores)

plt.ylabel('Correlação de Pearson')
for i in range(len(funcoes)):
    plt.text(i,  pearson[i], np.round(pearson[i],3), ha = 'center', color='black')
plt.xlabel('Funções de escore')
plt.xticks(rotation=70)
plt.title('Gráfico de Correlação de Pearson das funções de score')

#plt.show()
plt.savefig(salvar_path+'Pearson.png', format='png', bbox_inches='tight')
plt.clf()


for i in range(len(funcoes)):
    x = dataset[funcoes[i]]
    y = dataset['valor real']

    # Ajuste linear usando polyfit
    slope, intercept = np.polyfit(x, y, 1)

    # Criar a linha reta
    line = slope *y + intercept

    # Criar o gráfico de dispersão
    plt.scatter(x, y)

    # Adicionar a linha reta ao gráfico
    plt.plot(y, line, color='red', label='Equação linear dos dados reais')

    # Adicionar rótulos e legenda
    plt.xlabel('Dados preditos pela função de escore '+funcoes[i])
    plt.ylabel('Dados Reais')
    #plt.legend('Relação entre dados reais e os dados da função de escore '+funcoes[i])

    # Exibir o gráfico
    plt.savefig(salvar_path+funcoes[i]+'.png', format='png', bbox_inches='tight')
    plt.clf()




funcoes = list(dataset.columns[:-1])
pesos=[]
# Exemplo de valores reais e previstos
for coluna in funcoes:
  valores_reais = dataset[coluna]
  valores_previstos = dataset['valor real']

  # Calcular o MSE (Erro Quadrático Médio)
  mse = mean_squared_error(valores_reais, valores_previstos)

  # Calcular o RMSE
  rmse = np.sqrt(mse)
  #print(coluna,rmse)
  pesos.append(abs(rmse-1))
soma_pesos = sum(pesos)

pesos_normalizados = [peso / soma_pesos for peso in pesos]
#print(colunas)
#print("Pesos normalizados:", pesos_normalizados)

predicao_ponderada = []
for i in range(len(dataset)):
  soma = 0
  for j in range(len(pesos_normalizados)):
    soma = soma+pesos_normalizados[j]*dataset[funcoes[j]][i]
  predicao_ponderada.append(soma)

#print(resultado_predicao_ponderada)
resultado_predicao_ponderada = np.sqrt(mean_squared_error(predicao_ponderada, dataset['valor real']))



tabela_pesos = {
    'Funções de escore':  funcoes,
    'Pesos': pesos_normalizados,
}

# Criar DataFrame
tabela_pesos = pd.DataFrame(tabela_pesos)
fig, ax = plt.subplots(figsize=(10, 5)) 
ax.axis('off')
ax.table(cellText=tabela_pesos.values, colLabels=tabela_pesos.columns, cellLoc='center', loc='center')

plt.savefig(salvar_path+'pesos.png', format='png', bbox_inches='tight')
plt.clf()







X = dataset.drop(['valor real'],axis=1)

y = dataset['valor real']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo Random Forest
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Treinar o modelo
random_forest.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_predicao = random_forest.predict(X)

# Avaliar o desempenho usando métricas (RMSE neste exemplo)
rmse = np.sqrt(mean_squared_error(y_predicao, y))
print(f"Sem normalizacao RANDOM FOREST (RMSE): {rmse}")

scaler = MinMaxScaler()

X = dataset.drop(['valor real'],axis=1)
X = scaler.fit_transform(X)
y = dataset['valor real']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o modelo Random Forest
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Treinar o modelo
#random_forest.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
#y_pred = random_forest.predict(X)

# Avaliar o desempenho usando métricas (RMSE neste exemplo)
resultado_random_forest = np.sqrt(mean_squared_error(y_predicao, y))
previsao_random_forest = y_predicao
print(f"normalizado RANDOM FOREST (RMSE): {resultado_random_forest}")












X = dataset.drop(['valor real'],axis=1)

y = dataset['valor real']

# Divida os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialize o modelo XGBoost
xgboost_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0.5,
              importance_type='gain', interaction_constraints='',# num_class=2,
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1,
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, subsample=1, #scale_pos_weight="<class 'float'>",
              tree_method='exact', validate_parameters=1)

# Treine o modelo
xgboost_model.fit(X_train, y_train)

# Faça previsões no conjunto de teste
previsoes = xgboost_model.predict(X)

# Avalie o modelo
rmse = np.sqrt(mean_squared_error(y, previsoes))
print(f"sem normalização XGBOOST (RMSE): {rmse}")


X = dataset.drop(['valor real'],axis=1)
X = scaler.fit_transform(X)
y = dataset['valor real']

# Divida os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Treine o modelo
#modelo.fit(X_train, y_train)

# Faça previsões no conjunto de teste
#previsoes = modelo.predict(X)

# Avalie o modelo
resultado_xgboost = np.sqrt(mean_squared_error(previsoes,y))
previsoes_xgboost = previsoes
print(f"normalizado XGBOOST (RMSE): {resultado_xgboost}")























funcoes = list(dataset.columns[:-1])
resultado_rmse = []
for i in range(len(funcoes)):
    #print(funcoes[i], mean_squared_error(dataset[funcoes[i]], dataset['valor real']))
    resultado_rmse.append(np.sqrt(mean_squared_error(dataset[funcoes[i]], dataset['valor real'])))
resultado_rmse.append(resultado_random_forest)
resultado_rmse.append(resultado_xgboost)
resultado_rmse.append(resultado_predicao_ponderada)
funcoes.append('Ensemble com Random Forest')
funcoes.append('Ensemble com XGBoost')
funcoes.append('Função ponderada')
plt.bar(funcoes, resultado_rmse, color=paleta_cores)

plt.ylabel('RMSE')
for i in range(len(funcoes)):
    plt.text(i,  resultado_rmse[i]/2, np.round(resultado_rmse[i],3), ha = 'center', color='black')
plt.xlabel('Funções de escore')
plt.xticks(rotation=70)
plt.title('Gráfico comparativo de RMSE')

#plt.show()
plt.savefig(salvar_path+'Comparativo_dos_modelos_RMSE.png', format='png', bbox_inches='tight')

plt.clf()



funcoes = list(dataset.columns[:-1])
pearson = []
for i in range(len(funcoes)):
    pearson.append(dataset[funcoes[i]].corr(dataset['valor real']))
#print(previsao_random_forest,dataset['valor real'])
pearson.append(np.corrcoef(previsao_random_forest, dataset['valor real'])[0, 1])
pearson.append(np.corrcoef(previsoes_xgboost, dataset['valor real'])[0, 1])
pearson.append(np.corrcoef(predicao_ponderada, dataset['valor real'])[0, 1])
funcoes.append('Ensemble com Random Forest')
funcoes.append('Ensemble com XGBoost')
funcoes.append('Função ponderada')

plt.bar(funcoes, pearson, color=paleta_cores)

plt.ylabel('Correlação de Pearson')
for i in range(len(funcoes)):
    plt.text(i,  pearson[i], np.round(pearson[i],3), ha = 'center', color='black')
plt.xlabel('Funções de escore')
plt.xticks(rotation=70)
plt.title('Gráfico de Correlação de Pearson das funções de score e modelos')

plt.savefig(salvar_path+'Comparativo_dos_modelos_Pearson.png', format='png', bbox_inches='tight')
plt.clf()

#print('previsoes_xgboost', previsoes_xgboost)
#print('previsao_random_forest', previsao_random_forest)




x = previsoes_xgboost
y = dataset['valor real']

# Ajuste linear usando polyfit
slope, intercept = np.polyfit(x, y, 1)

# Criar a linha reta
line = slope *y + intercept

# Criar o gráfico de dispersão
plt.scatter(x, y)

# Adicionar a linha reta ao gráfico
plt.plot(y, line, color='red', label='Equação linear dos dados reais')

# Adicionar rótulos e legenda
plt.xlabel('Dados preditos pelo xgboost')
plt.ylabel('Dados Reais')
#plt.legend('Relação entre dados reais e os dados do xgboost')

# Exibir o gráfico
plt.savefig(salvar_path+'previsoes_xgboost.png', format='png', bbox_inches='tight')
plt.clf()





x = previsao_random_forest
y = dataset['valor real']

# Ajuste linear usando polyfit
slope, intercept = np.polyfit(x, y, 1)

# Criar a linha reta
line = slope *y + intercept

# Criar o gráfico de dispersão
plt.scatter(x, y)

# Adicionar a linha reta ao gráfico
plt.plot(y, line, color='red', label='Equação linear dos dados reais')

# Adicionar rótulos e legenda
plt.xlabel('Dados preditos pelo RF')
plt.ylabel('Dados Reais')
#plt.legend('Relação entre dados reais e os dados do xgboost')

# Exibir o gráfico
plt.savefig(salvar_path+'previsoes_rf.png', format='png', bbox_inches='tight')
plt.clf()




x = predicao_ponderada
y = dataset['valor real']

# Ajuste linear usando polyfit
slope, intercept = np.polyfit(x, y, 1)

# Criar a linha reta
line = slope *y + intercept

# Criar o gráfico de dispersão
plt.scatter(x, y)

# Adicionar a linha reta ao gráfico
plt.plot(y, line, color='red', label='Equação linear dos dados reais')

# Adicionar rótulos e legenda
plt.xlabel('Dados preditos pela função de ponderação')
plt.ylabel('Dados Reais')
#plt.legend('Relação entre dados reais e os dados da função de ponderação')

# Exibir o gráfico
plt.savefig(salvar_path+'previsoes_ponderada.png', format='png', bbox_inches='tight')
plt.clf()





#print(dataset)





dataset = pd.read_csv("/home/balboni/scripts/dataset_normalizado.txt",sep=';')

tabela_resultados = {
    'Ensemble RF':  previsao_random_forest,
    'Ensemble XGboost': previsoes_xgboost,
    'Função ponderada': predicao_ponderada,
    'valor real': dataset['valor real'],
    'pdb': dataset['pdb'],
}

# Criar DataFrame
tabela_resultados = pd.DataFrame(tabela_resultados)

colunas = tabela_resultados.columns
tabela_melhores = pd.DataFrame(columns = colunas)
del(tabela_melhores['pdb'])
del(tabela_melhores['valor real'])
for _ in range(10):
    tabela_melhores.loc[len(tabela_melhores)] = ''


tabela_piores = tabela_melhores.copy()

for column in tabela_melhores:
    data_aux = tabela_resultados[[column, 'valor real','pdb']]
    #print(data_aux)
    data_aux['diff'] = abs(data_aux['valor real'] - data_aux[column])
    data_aux = data_aux.sort_values(by='diff').reset_index()
    #print(data_aux)
    for j in range(10):
        tabela_melhores[column][j] = data_aux['pdb'][j]
        tabela_piores[column][j] = data_aux['pdb'][len(data_aux)-j-1]
    #break

#print(tabela_melhores)
fig, ax = plt.subplots(figsize=(10, 5)) 
ax.axis('off')
ax.table(cellText=tabela_melhores.values, colLabels=tabela_melhores.columns, cellLoc='center', loc='center')

plt.savefig(salvar_path+'Melhores predições ensemble.png', format='png', bbox_inches='tight')
plt.clf()




#print(tabela_piores)
fig, ax = plt.subplots(figsize=(10, 5)) 
ax.axis('off')
ax.table(cellText=tabela_piores.values, colLabels=tabela_piores.columns, cellLoc='center', loc='center')

plt.savefig(salvar_path+'Piores predições ensemble.png', format='png', bbox_inches='tight')
plt.clf()
