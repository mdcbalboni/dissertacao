import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset = pd.read_csv('/home/balboni/scripts/todos_juntos.txt',sep=';')

dataset = dataset.dropna()
dataset = dataset[~dataset.map(lambda x: 'Erro' in str(x)).any(axis=1)].reset_index(drop=True)


print(dataset['Lin_F9'].max())
print(dataset)
for column in dataset.iloc[:,1:]:
    try:
        dataset = dataset.loc[dataset[column] != 'Erro'].reset_index(drop=True)
        dataset[column] = dataset[column].astype(float)
        dataset[column] = dataset[column].abs()
    except:
        #print(dataset[column])
        continue
#print( type(dataset[column][i]))
print(dataset['Lin_F9'].max())
df_normalized = pd.DataFrame(scaler.fit_transform(dataset.iloc[:,1:]), columns=dataset.iloc[:,1:].columns)
# Exibir DataFrame normalizado
#print(df_normalized)
df_normalized['pdb'] = dataset['pdb']
df_normalized.to_csv('dataset_normalizado.txt',sep=';',index=False)