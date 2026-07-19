import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

class FeatureExtractor:
    def __init__(self, config):
        """
        Initializes the extractor using parameters from the YAML config.
        """
        self.radius = config['features'].get('morgan_radius', 2)
        # Defaulting to 2048 if not specified to reduce hash collisions
        self.n_bits = config['features'].get('morgan_nbits', 2048) 
        self.method = config['features'].get('method', 'morgan_plus_descriptors')

    def extract(self, smiles):
        """
        Converts a SMILES string into a 1D numpy array of features.
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
        
        # High-resolution Morgan Fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=self.radius, nBits=self.n_bits
        )
        fp_array = np.array(fp)
        
        if self.method == 'morgan_only':
            return fp_array
            
        # Global RDKit Descriptors
        descriptors = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol)
        ])
        
        return np.concatenate((fp_array, descriptors))