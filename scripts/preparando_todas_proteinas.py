import os
#!/usr/bin/python


caminho_inicial = '/home/balboni/'
caminho_base_de_dados = caminho_inicial + 'dados/v2020-other-PL/'


caminhos = [os.path.join(caminho_base_de_dados, nome) for nome in os.listdir(caminho_base_de_dados)]
lista_pdb = []
for dado in caminhos:
	lista_pdb.append(dado[-4:])
#print(lista_pdb)
#lista_pdb = ['1w8l']
for dado in lista_pdb:
	try:
		try:
			dados = open(caminho_base_de_dados+dado+'/'+dado+'_protein.pdbqt','r')
			continue
		except:
			pass
		dados = open(caminho_base_de_dados+dado+'/'+dado+'_protein.pdb','r')
		dados = dados.readlines()
		novo_dado = []
		saida = open(caminho_base_de_dados+dado+'/'+dado+'_protein_limpo.pdb', 'w')
		for dado_limpo in dados:
			if 'ATOM' in dado_limpo:
		#		novo_dado.append(dado)
				saida.write(dado_limpo)
		caminho_proteina = str(caminho_base_de_dados) + dado+'/'
		#os.system("obabel -i  pdb " + caminho_proteina+dado+'_protein_H.pdb -o pdbqt -O ' + caminho_proteina+dado+'_protein.pdbqt')
		#print(caminho_inicial + 'MGLTools-1.5.7/bin/pythonsh ' + caminho_inicial+ 'MGLTools-1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r ' + caminho_proteina+dado+'_protein.pdb -o pdbqt -O ' + caminho_proteina+dado+'_protein.pdbqt')
		print(caminho_inicial + 'mgltools/bin/pythonsh ' + caminho_inicial+ 'mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r ' + caminho_proteina+dado+'_protein_limpo.pdb -o ' + caminho_proteina+dado+'_protein.pdbqt')
		#break
		os.system(caminho_inicial + 'mgltools/bin/pythonsh ' + caminho_inicial+ 'mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r ' + caminho_proteina+dado+'_protein_limpo.pdb -o ' + caminho_proteina+dado+'_protein.pdbqt')
	except:
		continue

#/path/to/vina --receptor 1iep_protein.pdbqt --ligand 1iep_ligand.pdbqt --center_x 12.34 --center_y 54.45 --center_z 22.13 --cpu 2 --exhaustiveness 4 --size_x 12.98 --size_y 10.74 --size_z 15.79


#/home/balboni/MGLTools-1.5.7/bin/pythonsh /home/balboni/MGLTools-1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor.py -r /home/balboni/nova_metodologia/teste/1iep/1iep_protein.pdb -o /home/balboni/nova_metodologia/teste/1iep/1iep_protein.pdbqt
