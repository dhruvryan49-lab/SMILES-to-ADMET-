import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import os

def fetch_egfr_data():
    print("1. Querying ChEMBL API for EGFR target...")
    target = new_client.target
    target_query = target.search('EGFR')
    targets = pd.DataFrame.from_dict(target_query)
    
    # Get the exact CHEMBL ID for human EGFR
    egfr_target = targets[(targets['target_type'] == 'SINGLE PROTEIN') & 
                          (targets['tax_id'] == 9606)].iloc[0]
    chembl_id = egfr_target['target_chembl_id']
    print(f"   Found Target: {egfr_target['pref_name']} ({chembl_id})")

    print("\n2. Downloading raw bioactivity data (This will take a few minutes)...")
    activity = new_client.activity
    # Filter for exact IC50 measurements
    res = activity.filter(target_chembl_id=chembl_id)\
                  .filter(standard_type="IC50")\
                  .filter(relation="=")\
                  .filter(standard_units="nM")
                  
    df = pd.DataFrame.from_dict(res)
    print(f"   Downloaded {len(df)} raw records.")

    print("\n3. Cleaning data and calculating pIC50...")
    # Drop rows missing crucial data
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])
    
    # Remove extreme outliers and negative concentrations (API artifacts)
    df = df[(df['standard_value'] > 0) & (df['standard_value'] < 1e8)]

    # Calculate pIC50: pIC50 = 9 - log10(IC50_nM)
    df['pIC50'] = 9.0 - np.log10(df['standard_value'])
    
    # Keep only what the GNN needs
    clean_df = df[['canonical_smiles', 'pIC50']].copy()
    clean_df.rename(columns={'canonical_smiles': 'Drug', 'pIC50': 'Y'}, inplace=True)
    
    # Average duplicate SMILES (different labs testing the same drug)
    clean_df = clean_df.groupby('Drug').mean().reset_index()
    print(f"   Final dataset contains {len(clean_df)} unique molecules.")

    # Save to data directory
    os.makedirs('data', exist_ok=True)
    out_path = 'data/chembl_egfr.csv'
    clean_df.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Saved clean dataset to {out_path}")

if __name__ == "__main__":
    fetch_egfr_data()