import os
#!/usr/bin/python
import subprocess
import re
import pandas as pd



caminho_inicial = '/home/balboni/'
caminho_base_de_dados = caminho_inicial + 'dados/v2020-other-PL/'
caminho_saida = caminho_inicial + 'resultados/'
nome_dataset = caminho_inicial + 'resultado_general.txt'


caminhos = [os.path.join(caminho_base_de_dados, nome) for nome in os.listdir(caminho_base_de_dados)]
lista_pdb = []
for dado in caminhos:
	lista_pdb.append(dado[-4:])
#print(lista_pdb)

#todos_pdb = pd.read_csv(nome_dataset, sep = ',')
todos_pdb_txt = open(nome_dataset, 'r')
todos_pdb_txt = todos_pdb_txt.readlines()

todos_pdb = pd.DataFrame(columns = ['pdb','valor real'])
for dado in todos_pdb_txt[1:]:
	todos_pdb.loc[len(todos_pdb)] = [dado.split(',')[0],dado.split(',')[3]]
print(todos_pdb)
'''
for dado in lista_pdb:
	#try:
		aux_df = todos_pdb[todos_pdb['pdb']==dado]

		saida_dataset = saida_dataset.append(aux_df.iloc[0])
	#except:
		#print('Erro')
#print(saida_dataset)'''
todos_pdb.to_csv(caminho_saida+'resultado_experimental.txt',sep=';',index=False)
