import pandas as pd
from ecif import *
from os import listdir
import pickle

path_inicial = '/home/balboni/'
path_teste = path_inicial+'dados/v2020-other-PL/'
path_model = path_inicial+'funcoes/ECIF/ECIF6_GBT_model.pkl'
#print(path_teste)
Ligands = [path_teste+x for x in listdir(path_teste) if x[-3:] == "sdf"]
Ligands.sort()
Proteins = [path_teste+x for x in listdir(path_teste) if x[-3:] == "pdb"]
Proteins.sort()
#Ligands = ["Example_Structures/"+x for x in listdir("Example_Structures/") if x[-3:] == "sdf"]
Ligands = []
Proteins = []
for x in listdir(path_teste):
    Ligands.append(path_teste+x+'/'+x+'_ligand.sdf')
    Proteins.append(path_teste+x+'/'+x+'_protein.pdb')
Names = [x for x in Ligands]



#print(Ligands, Proteins)
ECIF = []
for protein, ligand in zip(Proteins, Ligands):
    try:
        ECIF.append(GetECIF(protein, ligand, distance_cutoff=6.0)) #for protein, ligand in zip(Proteins, Ligands)]
    except:
        continue
ligand_descriptors = []
for x in Ligands:
    try:
        ligand_descriptors.append(GetRDKitDescriptors(x))
    except:
        continue    
        #ligand_descriptors = [GetRDKitDescriptors(x) for x in Ligands]

#print(ligand_descriptors)

Descriptors = pd.DataFrame(ECIF, columns=PossibleECIF).join(pd.DataFrame(ligand_descriptors, columns=LigandDescriptors))
#print(Descriptors.head())

model = pickle.load(open(path_model, 'rb'))
#print(model)
for i in range(len(Names)):
    Names[i] = Names[i].split('_')[0][-4:]
Results = pd.DataFrame(Names, columns=["pdb"]).join(pd.DataFrame(model.predict(Descriptors), columns=["Saida ECIF"]))
print(Results)
Results.to_csv('saida_ECIF-GBT.txt',sep=';',index=False)