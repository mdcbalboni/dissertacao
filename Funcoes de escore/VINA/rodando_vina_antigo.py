import os
#!/usr/bin/python
import subprocess
import re


caminho_inicial = '/home/balboni/'
caminho_base_de_dados = caminho_inicial + 'dados/v2020-other-PL/'
vina_path = 'autodock_vina_1_1_2_linux_x86/bin/vina'
caminho_saida = caminho_inicial + 'resultados/'

caminhos = [os.path.join(caminho_base_de_dados, nome) for nome in os.listdir(caminho_base_de_dados)]
lista_pdb = []
for dado in caminhos:
	lista_pdb.append(dado[-4:])
print(lista_pdb)


saida = open(caminho_saida+'vina_saida.txt','w')
saida.write('pdb;score_vina\n')
for dado in lista_pdb:
	try:
		caminho_proteina = str(caminho_base_de_dados) + dado+'/'
		output = subprocess.check_output(caminho_inicial+vina_path+ ' --receptor ' + caminho_proteina+dado+'_protein.pdbqt --ligand ' + caminho_proteina+dado+'_ligand.pdbqt  --score_only', shell=True)
		output = output.decode("utf-8")
		index = output.index('Affinity: ')
		#print(output[index],index)
		index = index + 10
		resultado = []
		for i in range(index,index+15,1):
			#print(output[i])
			if output[i] == ' ':
				break
			resultado.append(output[i])
		resultado = ''.join(resultado)
		resultado = re.sub('[^0-9-.]', '', resultado)
		saida.write(dado+';'+str(resultado)+'\n')
		#break
	except:
		saida.write(dado+';Erro\n')
#/home/balboni/Desktop/Programas/autodock_vina_1_1_2_linux_x86/bin/vina --receptor /home/balboni/nova_metodologia/teste/1iep/1iep_protein.pdbqt --ligand /home/balboni/nova_metodologia/teste/1iep/1iep_ligand.pdbqt --score_only

