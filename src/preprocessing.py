import pandas as pd

class Preprocessor:
    @staticmethod
    def clean_data(df, feature_column='Features'):
        """
        Removes rows where feature extraction failed (e.g., invalid SMILES).
        Future upgrades like SMOTE for class imbalance will go here.
        """
        initial_len = len(df)
        df_cleaned = df.dropna(subset=[feature_column]).copy()
        dropped = initial_len - len(df_cleaned)
        
        if dropped > 0:
            print(f"      [Preprocessor] Dropped {dropped} invalid molecules.")
            
        return df_cleaned