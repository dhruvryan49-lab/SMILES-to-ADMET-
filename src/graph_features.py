import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors

class GraphExtractor:
    def __init__(self, config=None):
        # We define the allowable atoms to one-hot encode them
        self.allowable_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']

    def _one_hot_encoding(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1] # Map unknown atoms to the last element
        return list(map(lambda s: x == s, allowable_set))

    def extract(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None # We will filter these out in the data loader
            
        # 1. Node Features (Atom properties)
        node_features = []
        for atom in mol.GetAtoms():
            # One-hot encode the atom type and its degree (number of bonds)
            atom_type = self._one_hot_encoding(atom.GetSymbol(), self.allowable_atoms)
            degree = self._one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5, 6])
            is_aromatic = [int(atom.GetIsAromatic())]
            
            # Combine into a single feature vector for this atom
            node_features.append(atom_type + degree + is_aromatic)
            
        x = torch.tensor(node_features, dtype=torch.float)

        # 2. Edge Indices & Features (Bonds)
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Encode bond type (Single, Double, Triple, Aromatic)
            bond_type = bond.GetBondTypeAsDouble()
            is_aromatic = int(bond.GetIsAromatic())
            
            # Molecules are undirected graphs, so we add both directions: i->j and j->i
            edge_indices += [[i, j], [j, i]]
            edge_attrs += [[bond_type, is_aromatic], [bond_type, is_aromatic]]

        if len(edge_indices) > 0:
            # PyG expects edge_index to be shape [2, num_edges]
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            # Fallback for single-atom molecules (like single ions)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        rings = Descriptors.RingCount(mol)
        fcsp3 = Descriptors.FractionCSP3(mol)
        
        # Normalize them roughly so they don't overpower the neural network gradients
        global_features = [mw/500, logp/5, tpsa/100, hbd/5, hba/10, rotb/10, rings/5, fcsp3]
        global_tensor = torch.tensor([global_features], dtype=torch.float)

        # 4. Construct the PyTorch Geometric Data object with the new global tensor
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_tensor)