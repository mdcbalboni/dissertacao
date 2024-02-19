import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
import pickle


ecif = pd.read_csv("Descriptors/ECIF_6.0.csv") # Load ECIF (Compressed File)
ligand_descriptors = pd.read_csv("Descriptors/RDKit_Descriptors.csv") # Load ligand descriptors
binding_data = pd.read_csv("Descriptors/BindingData.csv") # Load binding affinity data

# Merge descriptors
ecif = ecif.merge(ligand_descriptors, left_on="PDB", right_on="PDB")
ecif = ecif.merge(binding_data, left_on="PDB", right_on="PDB")
x_train = ecif[ecif["SET"] == "Train"][list(ecif.columns)[1:-2]]
y_train = ecif[ecif["SET"] == "Train"]["pK"]

x_test = ecif[ecif["SET"] == "Test"][list(ecif.columns)[1:-2]]
y_test = ecif[ecif["SET"] == "Test"]["pK"]

print(x_train.shape[0], x_test.shape[0])

RF = RandomForestRegressor(random_state=1206, n_estimators=500, n_jobs=8, oob_score=True, max_features=0.33)

roda_rf = 'n'
if roda_rf == 's':
    RF.fit(x_train,y_train)

    pickle.dump(RF, open("ECIF6_RF_model.pkl", 'wb'))

GBT = GradientBoostingRegressor(random_state=1206, n_estimators=20000, max_features="sqrt", max_depth=8, min_samples_split=3, learning_rate=0.005, subsample=0.7)
roda_GBT = 's'
if roda_GBT == 's':
    GBT.fit(x_train,y_train)

    pickle.dump(GBT, open("ECIF6_GBT_model.pkl", 'wb'))