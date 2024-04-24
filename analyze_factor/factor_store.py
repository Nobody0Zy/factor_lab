import pandas as pd
import def_factor_repo

class FactorStore:
    def __init__(self):
        self.factor_store_path = r'F:\factor_lab_res\factor_stock.h5'
        self.factor_store = pd.HDFStore(self.factor_store_path)
        self.factor_name_list = self.factor_store.keys()
        
    
    