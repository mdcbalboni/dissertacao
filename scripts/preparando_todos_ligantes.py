import os


caminho_inicial = '/home/balboni/'
caminho_base_de_dados = caminho_inicial + 'dados/v2020-other-PL/'

#lista_pdb = ['1a0q','1a0t','1a1b','1a1c','1a2c','1a3e','1a4g','1a4h','1a4m','1a4q','1a5g','1a5h','1a5v']

caminhos = [os.path.join(caminho_base_de_dados, nome) for nome in os.listdir(caminho_base_de_dados)]
lista_pdb = []
for dado in caminhos:
	lista_pdb.append(dado[-4:])
#print(lista_pdb)


for dado in lista_pdb:
	caminho_ligante = str(caminho_base_de_dados) + dado+'/'
	os.system("obabel -i  sdf " + caminho_ligante+dado+'_ligand.sdf -o pdbqt -O ' + caminho_ligante+dado+'_ligand.pdbqt')
