import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs 

class FeatureExtractor:
    def __init__(self, config=None):
        # We catch the config dictionary passed by main.py so it doesn't overwrite our radius
        self.radius = 2
        self.n_bits = 2048
        # Total length will be 2048 (Morgan) + 8 (Descriptors) = 2056
        self.total_features = self.n_bits + 8

    def extract(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # If RDKit cannot parse the SMILES, return a zero vector
            return np.zeros(self.total_features, dtype=np.float32)
            
        # 1. Morgan Fingerprint (Pre-allocated for memory safety)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        fp_array = np.zeros((self.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        # 2. RDKit 2D Descriptors (Global Physicochemical Properties)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        rings = Descriptors.RingCount(mol)
        fcsp3 = Descriptors.FractionCSP3(mol) # Ratio of sp3 carbons (3D structure proxy)
        
        # Group into a float32 array
        desc_array = np.array([mw, logp, tpsa, hbd, hba, rotb, rings, fcsp3], dtype=np.float32)
        
        # Concatenate local and global features
        return np.concatenate([fp_array, desc_array])