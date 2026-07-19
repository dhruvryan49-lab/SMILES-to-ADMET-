import pandas as pd
from tdc.single_pred import ADME, Tox

class ADMETDataLoader:
    def __init__(self, config, endpoint_name):
        self.config = config
        self.endpoint_name = endpoint_name
        self.endpoint_config = config['endpoints'][endpoint_name]

    def get_data_splits(self):
        data_config = self.endpoint_config['data']
        split_config = self.endpoint_config['split']
        
        category = data_config['category']
        dataset_name = data_config['dataset_name']
        
        print(f"[{self.endpoint_name}] Fetching {dataset_name} from TDC ({category})...")

        if category == 'ADME':
            data = ADME(name=dataset_name)
        elif category == 'Tox':
            data = Tox(name=dataset_name)
        else:
            raise ValueError(f"Unknown category in config: {category}")

        print(f"[{self.endpoint_name}] Executing {split_config['method']} split...")
        split = data.get_split(
            method=split_config['method'], 
            frac=[
                split_config['train_frac'],
                split_config['val_frac'],
                split_config['test_frac']
            ]
        )
        
        return split['train'].copy(), split['valid'].copy(), split['test'].copy()