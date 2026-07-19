from tdc.single_pred import ADME, Tox

class ADMETDataLoader:
    def __init__(self, config, endpoint_name):
        self.config = config
        self.endpoint_name = endpoint_name
        self.endpoint_config = config['endpoints'][endpoint_name]

    def get_data_splits(self):
        data_config = self.endpoint_config['data']
        provider = data_config['provider']
        category = data_config['category']
        dataset_name = data_config['dataset_name']
        
        # Pull the target column to use as the specific label for multi-label datasets
        label_name = data_config.get('label_name')
        
        print(f"   [{self.endpoint_name}] Fetching {dataset_name} from {provider} ({category})...")
        
        if provider == 'TDC':
            if category == 'ADME':
                data = ADME(name=dataset_name, label_name=label_name)
            elif category == 'Tox':
                data = Tox(name=dataset_name, label_name=label_name)
            else:
                raise ValueError(f"Unknown category: {category}")
                
            split_config = self.endpoint_config['split']
            
            # Fetch the split dictionary from TDC
            splits = data.get_split(
                method=split_config['method'], 
                frac=[split_config['train_frac'], split_config['val_frac'], split_config['test_frac']]
            )
            
            # Extract and return the actual Pandas DataFrames
            return splits['train'], splits['valid'], splits['test']
        else:
            raise ValueError(f"Unknown data provider: {provider}")