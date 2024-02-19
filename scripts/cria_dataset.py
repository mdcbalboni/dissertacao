import os
import pandas as pd
#!/usr/bin/python


caminho_inicial = '/home/balboni/'
caminho_base_de_dados = caminho_inicial + 'resultados/'

caminhos = [os.path.join(caminho_base_de_dados, nome) for nome in os.listdir(caminho_base_de_dados)]
lista_pdb = []
lista_pdb.append('/home/balboni/resultados/resultado_experimental.txt')

for dado in caminhos:
	if 'grafico' in dado or 'experimental' in dado:
		continue
	lista_pdb.append(dado)
print(lista_pdb)
resultado_unidos = pd.read_csv(lista_pdb[0],sep=';')

for dado in lista_pdb[1:]:
	try:
		print(dado)
		df_add = pd.read_csv(dado,sep=';')
		resultado_unidos = resultado_unidos.merge(df_add, how='inner', on='pdb')
	except:
		continue

resultado_unidos = resultado_unidos.dropna()
label = resultado_unidos['valor real']
del(resultado_unidos['valor real'])
resultado_unidos['valor real'] = label
print(resultado_unidos)
resultado_unidos.to_csv('todos_juntos.txt',sep=';',index=False)
