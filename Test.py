import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def main():
    print("Loading a sample of the ChEMBL EGFR dataset...")
    # Load the cleaned ChEMBL dataset you mined earlier
    df = pd.read_csv('data/chembl_egfr.csv')
    print("Columns in dataset:", df.columns.tolist())
    
    # Sample 2,000 molecules for a quick baseline test
    df = df.sample(n=2000, random_state=42)
    
    print(f"Extracting 1D Morgan Fingerprints for {len(df)} molecules...")
    X = np.array([smiles_to_fingerprint(s) for s in df['Drug']])
    y = df['Y'].values
    
    # Standard 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Classical XGBoost Regressor...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating on the test split...")
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n==================================================")
    print(" XGBOOST CHEMBL EGFR BASELINE ")
    print("==================================================")
    print(f" Test RMSE: {rmse:.3f}")
    print(f" Test R-Squared: {r2:.3f}")
    print("==================================================")

if __name__ == "__main__":
    main()