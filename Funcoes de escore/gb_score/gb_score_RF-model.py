import os
import sys
from glob import glob
from pathlib import Path
from shutil import copyfile
import argparse
import numpy as np
import pandas as pd
import joblib
from joblib import dump, load
import matplotlib.pyplot as plt
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
#from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import spatial
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor



#import generate_features

def preprocessing(data, var_threshold=0.01, corr_threshold=0.95):

   # data = data.loc[:, data.var(axis=0) > var_threshold]

  #  corr_matrix = data.corr().abs()
  #  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  #  to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
   # data = data.drop(to_drop, axis=1)
    
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std

    return data, mean, std

atom_types = {
    "OE1": "O",
    "HB1": "H",
    "SD": "S",
    "HE": "H",
    "1HD1": "H",
    "1HG1": "H",
    "2HD2": "H",
    "1HD2": "H",
    "CE3": "C",
    "OH": "O",
    "CZ": "C",
    "HG2": "H",
    "HN2": "H",
    "NZ": "N",
    "HN1": "H",
    "3HD2": "H",
    "CD2": "C",
    "2HH2": "H",
    "HH2": "H",
    "O": "O",
    "2HD1": "H",
    "ND1": "N",
    "HH": "H",
    "1HE2": "H",
    "HB": "H",
    "NH2": "N",
    "3HG1": "H",
    "ND2": "N",
    "CZ3": "C",
    "HA2": "H",
    "OG": "O",
    "CG2": "C",
    "CE": "C",
    "SG": "S",
    "NE": "N",
    "CG": "C",
    "CB": "C",
    "HG1": "H",
    "NH1": "N",
    "2HE2": "H",
    "3HD1": "H",
    "1HH2": "H",
    "HD2": "H",
    "HD1": "H",
    "NE1": "N",
    "HB2": "H",
    "HA": "H",
    "3HG2": "H",
    "HN3": "H",
    "HE1": "H",
    "CD": "C",
    "HZ3": "H",
    "OD1": "O",
    "N": "N",
    "H": "H",
    "HA1": "H",
    "2HH1": "H",
    "NE2": "N",
    "CE2": "C",
    "C": "C",
    "OD2": "O",
    "2HG1": "H",
    "CD1": "C",
    "HE3": "H",
    "1HH1": "H",
    "2HG2": "H",
    "HB3": "H",
    "CE1": "C",
    "OXT": "O",
    "CH2": "C",
    "1HG2": "H",
    "HZ1": "H",
    "OG1": "O",
    "HZ2": "H",
    "CA": "C",
    "CG1": "C",
    "HE2": "H",
    "CZ2": "C",
    "OE2": "O",
    "HG": "H",
    "HZ": "H",
}

amino_acid_groups = [
    ["ARG", "LYS", "ASP", "GLU"],
    ["GLN", "ASN", "HIS", "SER", "THR", "CYS"],
    ["TRP", "TYR", "MET"],
    ["ILE", "LEU", "PHE", "VAL", "PRO", "GLY", "ALA"],
]

elements = ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]


def generate_columns_name(elements, amino_acid_groups):

    columns_name = {}

    i = 0

    for ligand_element in elements:

        for protein_element in elements:

            for index, amino_acid_group in enumerate(amino_acid_groups):

                columns_name[i] = (
                    ligand_element + "_" + protein_element + "_" + str(index)
                )

                i += 1

    return columns_name


def ligand_specific_element_coordinates(ligand, element, complex=False):

    if complex:
        coordinates = ligand[ligand["element_symbol"] == element].loc[
            :, ["x_coord", "y_coord", "z_coord"]
        ]

    else:
        coordinates = ligand[ligand["element_symbol"] == element].loc[
            :, ["x", "y", "z"]
        ]

    return coordinates.to_numpy()


def protein_specific_element_coordinates(protein, element, amino_acid_group):

    protein = protein[protein["residue_name"].isin(amino_acid_group)]

    coordinates = protein[protein["element_symbol"] == element].loc[
        :, ["x_coord", "y_coord", "z_coord"]
    ]

    return coordinates.to_numpy()


def weighting_sum(distances, cutoff, exp):

    selected_distances = distances[distances < cutoff]

    feature = sum(list(map(lambda x: 1.0 / (x ** exp), selected_distances)))

    return feature


def generate_features(path, exp=2, complex=False, filename="file.csv"):
    
    data = {}

    entries = Path(path)

    numbers = len(os.listdir(path))


    for num, entry in enumerate(entries.iterdir()):


        features = []
	
        if complex:
            complex_file = glob(str(entry))

            pdf = PandasPdb().read_pdb(complex_file[0])

            pmol = pdf.df["HETATM"]

            ppdb = pdf.df["ATOM"]
            print(ppdb,pmol)
            
            ppdb["element_symbol"] = ppdb["atom_name"].map(atom_types)

        else:
        
            
            ligand_file = glob(str(entry) + "//*.mol2")
            print(ligand_file)
            protein_file = glob(str(entry) + "//*.pdb")
            
            try:
            	pmol = PandasMol2().read_mol2(ligand_file[0])
            	
            except Exception:
            
            	print(f"Error in file structure {str(entry.name)}")
            	
            	continue

            pmol = pmol.df

            pmol["element_symbol"] = pmol["atom_type"].apply(lambda x: x.split(".")[0])

            ppdb = PandasPdb().read_pdb(protein_file[0])

            ppdb = ppdb.df["ATOM"]

            ppdb["element_symbol"] = ppdb["atom_name"].map(atom_types)

        for ligand_element in elements:

            for protein_element in elements:

                for amino_acid_group in amino_acid_groups:

                    ligand_coords = ligand_specific_element_coordinates(
                        pmol, ligand_element, complex
                    )

                    protein_coords = protein_specific_element_coordinates(
                        ppdb, protein_element, amino_acid_group
                    )

                    distances = spatial.distance.cdist(
                        ligand_coords, protein_coords
                    ).ravel()

                    features.append(weighting_sum(distances, 12.0, exp))

        if complex:

            name = entry.name

        else:
            name = entry.name

        data[name] = features

    columns_name = generate_columns_name(elements, amino_acid_groups)

    df = pd.DataFrame(data).transpose().rename(columns=columns_name)

    return df.to_csv(filename,sep=';')



path_inicial = '/home/balboni/funcoes/'
path_dataset = '/home/balboni/dados/v2020-other-PL/'
gerar_feature = 'n' 
if gerar_feature == 's':
    generate_features(path_dataset, 2, False, 'dataset_gb_score.csv')


dataset_gerado = '/home/balboni/funcoes/gb_score_RF-model/dataset_gb_score.csv'
dataset_para_predicao = pd.read_csv(dataset_gerado,sep=';')#,index_col=0)
print(dataset_para_predicao)
pdbs = dataset_para_predicao['Unnamed: 0']
del(dataset_para_predicao['Unnamed: 0'])
dataset_normalizado, mean, std = preprocessing(dataset_para_predicao, var_threshold=0.01, corr_threshold=0.95)
path_saved_model = "/home/balboni/funcoes/gb_score/regression_model.joblib"
gb_reg = joblib.load(path_saved_model)
#x_features = preprocessor.fit(x_train_up).get_feature_names_out()
dataset_normalizado = dataset_normalizado[gb_reg.feature_names_in_]
dataset_normalizado = dataset_normalizado.fillna(0)
y_pred = gb_reg.predict(dataset_normalizado)
resultado = pd.DataFrame(columns = ['pdbs','rb_score_RF'])
for i in range(len(y_pred)):
    resultado.loc[len(resultado)] = [pdbs[i],y_pred[i]]
resultado.to_csv('predicao_gb_score_RF-model.txt',sep=';',index=False)