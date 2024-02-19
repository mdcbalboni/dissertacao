import os
#!/usr/bin/python
import subprocess
import re


caminho_inicial = '/home/balboni/'
caminho_base_de_dados = caminho_inicial + 'dados/v2020-other-PL/'
funcao_path = 'funcoes/Convex/Convex-PL'
caminho_saida = caminho_inicial + 'resultados/'

caminhos = [os.path.join(caminho_base_de_dados, nome) for nome in os.listdir(caminho_base_de_dados)]
lista_pdb = []
for dado in caminhos:
	lista_pdb.append(dado[-4:])
print(lista_pdb)

saida = open(caminho_saida+'Convex-PL-0.5_saida.txt','w')
saida.write('pdb;score_Convex-PL\n')
for dado in lista_pdb:
	try:
		caminho_proteina = str(caminho_base_de_dados) + dado+'/'
		output = os.popen(caminho_inicial+funcao_path+ ' --receptor ' + caminho_proteina+dado+'_protein.pdb --ligand ' + caminho_proteina+dado+'_ligand.mol2 --mol2 --regscore').read()
		#print(output)
		index = output.index('score = ')
		index = index + 10
		resultado = []
		for i in range(index,index+10,1):
			#print(output[i])
			if output[i] == ' ':
				break
			resultado.append(output[i])
		resultado = ''.join(resultado)
		resultado = re.sub('[^0-9-.]', '', resultado)
		print(resultado)
		saida.write(dado+';'+str(resultado)+'\n')
		#break
	except:
		saida.write(dado+';Erro\n')
#/home/balboni/Desktop/Programas/autodock_vina_1_1_2_linux_x86/bin/vina --receptor /home/balboni/nova_metodologia/teste/1iep/1iep_protein.pdbqt --ligand /home/balboni/nova_metodologia/teste/1iep/1iep_ligand.pdbqt --score_only

