import torch
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_df = None
        self.processed_data = None
    
    def load_and_process_all_timepoints(self, timepoints):
        data_dict = {}
        skip_keys = {'plot_id', 'irrigation', 'irrigation_labels', 'coordinates', 'yield'}
        stack_keys = {'vegetation', 'cwsi'}
        
        for i, tp in enumerate(timepoints):
            tp_data = self.load_and_process(tp)
            for key, value in tp_data.items():
                if key in skip_keys:
                    # For constant keys, store the value only once.
                    if key not in data_dict:
                        data_dict[key] = value
                    continue
    
                if key in stack_keys:
                    # For the first timepoint, add a new axis.
                    if key not in data_dict:
                        data_dict[key] = np.expand_dims(value, axis=1)
                    else:
                        # For subsequent timepoints, also expand dims before concatenating.
                        data_to_add = np.expand_dims(value, axis=1)
                        data_dict[key] = np.concatenate((data_dict[key], data_to_add), axis=1)
        return data_dict
                        
                    

    def load_and_process(self, timepoint):
        """Load and process data for specific timepoint"""
        # Load raw data
        with open(self.file_path, "rb") as f:
            self.raw_df = pickle.load(f)

        # Filter timepoint
        df_tp = self.raw_df[self.raw_df['timepoint'] == timepoint].copy()

        # Clean data
        df_tp = self._clean_data(df_tp)
        
        
        if "image_dataset" in self.file_path: 
            self._resize_cwsi(df_tp)
            return self._create_image_datadict(df_tp)
        elif "stat_dataset" in self.file_path:
            return self._create_stat_datadict(df_tp)
        else:
            print("Unsupported dataset type.")
            return None
        
    def _clean_data(self, df):
        """Handle missing values and columns"""
        # initial_count = len(df)
        df = df.dropna(subset=['Yield'])
        # print(f"Removed {initial_count - len(df)} samples with missing Yield")
        return df.drop(columns=['timepoint'])

    def _resize_cwsi(self, df):
        """Skip resizing CWSI to retain original dimensions"""
        # No resizing is applied
        df['CWSI'] = df['CWSI'].apply(lambda x: torch.from_numpy(x).float())

    def _create_image_datadict(self, df):
        """Create data dictionary with tensors"""
        # Vegetation indices (5 channels)
        veg_data = np.stack([
            np.stack(df['NDVI'], axis=0),
            np.stack(df['NDRE'], axis=0),
            np.stack(df['GCI'], axis=0),
            np.stack(df['MTCI'], axis=0),
            np.stack(df['EVI'], axis=0)
        ], axis=1)

        # CWSI data (1 channel, original size)
        cwsi_data = np.stack(df['CWSI'], axis=0)

        # Irrigation encoding
        irrig_encoder = OrdinalEncoder(categories=[['P33', 'P67', 'P100']])
        irrigation = irrig_encoder.fit_transform(df[['irrigation_trt']])

        return {
            'plot_id': df['plot_id'].values,
            'vegetation': veg_data,
            'cwsi': cwsi_data,
            'irrigation': irrigation,
            'yield': df['Yield'].values,
            'coordinates': df[['lat', 'lon']].values,
            'irrigation_labels': df['irrigation_trt'].values
        }
    
    def _create_stat_datadict(self, df):
        
        irrig_encoder = OrdinalEncoder(categories=[['P33', 'P67', 'P100']])
        irrigation = irrig_encoder.fit_transform(df[['irrigation_trt']])
        
        df['irrigation'] = irrigation
        
        df.rename(columns={'irrigation_trt': 'irrigation_labels'}, inplace=True)
        df.rename(columns={'Yield': 'yield'}, inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        
        return df
