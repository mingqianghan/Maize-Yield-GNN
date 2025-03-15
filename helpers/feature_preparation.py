import torch
import pickle
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

class DataProcessor:
    """
    A class to load, clean, and process dataset for yield prediction.

    This processor handles both image-based and statistical datasets by loading the raw data
    from a pickle file, filtering by timepoint, cleaning missing values, and creating a data
    dictionary with the necessary keys for further modeling.
    """
    def __init__(self, file_path):
        """
        Initialize the DataProcessor.

        Parameters:
            file_path (str): Path to the pickle file containing the raw dataset.
        """
        self.file_path = file_path
        self.raw_df = None

    def load_and_process_all_timepoints(self, timepoints):
        """
        Load and process the data for all specified timepoints.

        For keys that represent constant information (e.g., plot_id, irrigation, yield),
        the value is stored only once. For image-based data keys (e.g., vegetation, cwsi),
        the data is stacked along a new axis representing the time dimension.

        Parameters:
            timepoints (list): List of timepoint identifiers (e.g., ["R1", "R2"]).

        Returns:
            dict: A dictionary containing the processed data across all timepoints.
        """
        data_dict = {}
        # Keys that remain constant over time.
        skip_keys = {'plot_id', 'irrigation', 'irrigation_labels', 'coordinates', 'yield'}
        # Keys for which the data should be stacked along the time dimension.
        stack_keys = {'vegetation', 'cwsi'}
        
        for i, tp in enumerate(timepoints):
            tp_data = self.load_and_process(tp)
            for key, value in tp_data.items():
                if key in skip_keys:
                    # Store constant values only once.
                    if key not in data_dict:
                        data_dict[key] = value
                    continue
                if key in stack_keys:
                    # For the first timepoint, add a new axis for time.
                    if key not in data_dict:
                        data_dict[key] = np.expand_dims(value, axis=1)
                    else:
                        # For subsequent timepoints, expand dims and concatenate along the time axis.
                        data_to_add = np.expand_dims(value, axis=1)
                        data_dict[key] = np.concatenate((data_dict[key], data_to_add), axis=1)
        return data_dict

    def load_and_process(self, timepoint):
        """
        Load and process data for a specific timepoint.

        This function reads the raw pickle file, filters the data to the specified timepoint,
        cleans the data by removing samples with missing yield values, and then processes the
        data based on the dataset type (image-based or statistical).

        Parameters:
            timepoint (str): The timepoint identifier (e.g., "R1").

        Returns:
            dict or DataFrame: A processed data dictionary for image datasets or a DataFrame for statistical datasets.
        """
        # Load raw data from file.
        with open(self.file_path, "rb") as f:
            self.raw_df = pickle.load(f)

        # Filter data for the specified timepoint.
        df_tp = self.raw_df[self.raw_df['timepoint'] == timepoint].copy()

        # Clean data: drop rows with missing 'Yield' and remove the timepoint column.
        df_tp = self._clean_data(df_tp)
        
        # Process data based on dataset type.
        if "image_dataset" in self.file_path: 
            self._resize_cwsi(df_tp)
            return self._create_image_datadict(df_tp)
        elif "stat_dataset" in self.file_path:
            return self._create_stat_datadict(df_tp)
        else:
            print("Unsupported dataset type.")
            return None

    def _clean_data(self, df):
        """
        Clean the input DataFrame by handling missing values and unnecessary columns.

        Specifically, it drops samples with missing 'Yield' values and removes the 'timepoint' column.

        Parameters:
            df (pandas.DataFrame): DataFrame to be cleaned.

        Returns:
            pandas.DataFrame: Cleaned DataFrame.
        """
        df = df.dropna(subset=['Yield'])
        return df.drop(columns=['timepoint'])

    def _resize_cwsi(self, df):
        """
        Process the CWSI column to convert its contents into torch tensors.

        No resizing of dimensions is applied; this method only converts each CWSI value from a NumPy array
        into a PyTorch float tensor.

        Parameters:
            df (pandas.DataFrame): DataFrame containing the CWSI column.
        """
        df['CWSI'] = df['CWSI'].apply(lambda x: torch.from_numpy(x).float())

    def _create_image_datadict(self, df):
        """
        Create a data dictionary from an image-based DataFrame.

        Stacks vegetation indices into a multi-channel array, organizes CWSI data, encodes the irrigation treatment,
        and collects additional attributes such as plot_id, yield, coordinates, and irrigation labels.

        Parameters:
            df (pandas.DataFrame): Cleaned DataFrame for a specific timepoint.

        Returns:
            dict: Data dictionary containing image data and related attributes.
        """
        # Stack vegetation indices into a 5-channel array.
        veg_data = np.stack([
            np.stack(df['NDVI'], axis=0),
            np.stack(df['NDRE'], axis=0),
            np.stack(df['GCI'], axis=0),
            np.stack(df['MTCI'], axis=0),
            np.stack(df['EVI'], axis=0)
        ], axis=1)

        # Stack CWSI data (assumed to be single-channel).
        cwsi_data = np.stack(df['CWSI'], axis=0)

        # Encode irrigation treatments into ordinal values.
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
        """
        Create a processed DataFrame for a statistical dataset.

        This method encodes irrigation treatments, renames columns for consistency, and resets the index.

        Parameters:
            df (pandas.DataFrame): Cleaned DataFrame for a specific timepoint.

        Returns:
            pandas.DataFrame: Processed DataFrame with renamed and encoded columns.
        """
        # Encode the irrigation treatment.
        irrig_encoder = OrdinalEncoder(categories=[['P33', 'P67', 'P100']])
        irrigation = irrig_encoder.fit_transform(df[['irrigation_trt']])
        df['irrigation'] = irrigation
        
        # Rename columns for consistency.
        df.rename(columns={'irrigation_trt': 'irrigation_labels', 'Yield': 'yield'}, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df