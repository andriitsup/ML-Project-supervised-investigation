import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Dict, Tuple, List

def download_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Download and return a copy of the dataset.
    
    Args:
        raw_df (pd.DataFrame): The raw input dataframe.
        
    Returns:
        pd.DataFrame: A copy of the input dataframe.
    """
    return raw_df.copy()

def drop_unimportant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unimportant columns from the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        
    Returns:
        pd.DataFrame: The dataframe with unimportant columns dropped.
    """
    return df.drop(columns=['CustomerId', 'Surname'], axis=1)

def create_train_val_split(df: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create training and validation sets.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        target (pd.Series): The target series.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation dataframes.
    """
    return train_test_split(df, test_size=0.2, random_state=42, stratify=target)

def create_inputs_targets(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create inputs and targets for the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target column.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Inputs and targets dataframes.
    """
    input_cols = list(df.columns)[1:-1]
    inputs = df[input_cols].copy()
    targets = df[target_col].copy()
    return inputs, targets

def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        
    Returns:
        Tuple[List[str], List[str]]: Lists of numeric and categorical column names.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    return numeric_cols, categorical_cols

def scale_numeric_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, numeric_cols: List[str], scale: bool) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric features if the scale parameter is True.
    
    Args:
        train_inputs (pd.DataFrame): Training inputs.
        val_inputs (pd.DataFrame): Validation inputs.
        numeric_cols (List[str]): List of numeric column names.
        scale (bool): Whether to scale numeric features.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: Scaled training and validation inputs, and the scaler.
    """
    
    scaler = MinMaxScaler().fit(train_inputs[numeric_cols])
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler

def one_hot_encode_categorical_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode categorical features.
    
    Args:
        train_inputs (pd.DataFrame): Training inputs.
        val_inputs (pd.DataFrame): Validation inputs.
        categorical_cols (List[str]): List of categorical column names.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]: Encoded training and validation inputs, and the encoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    train_encoded = pd.DataFrame(encoder.transform(train_inputs[categorical_cols]), columns=encoded_cols, index=train_inputs.index)
    val_encoded = pd.DataFrame(encoder.transform(val_inputs[categorical_cols]), columns=encoded_cols, index=val_inputs.index)
    train_inputs = train_inputs.drop(columns=categorical_cols).join(train_encoded)
    val_inputs = val_inputs.drop(columns=categorical_cols).join(val_encoded)
    return train_inputs, val_inputs, encoder

def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder]:
    """
    Process the raw data and return the training and validation sets, along with
    input columns, scaler, and encoder.
    
    Args:
        raw_df (pd.DataFrame): The raw input dataframe.
        scaler_numeric (bool): Whether to scale numeric features.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder]:
        Processed training inputs, training targets, validation inputs, validation targets,
        input column names, scaler, and encoder.
    """
    bank_df = download_dataset(raw_df)
    bank_df = drop_unimportant_columns(bank_df)
    target = bank_df['Exited']
    
    train_df, val_df = create_train_val_split(bank_df, target)
    
    train_inputs, train_targets = create_inputs_targets(train_df, 'Exited')
    val_inputs, val_targets = create_inputs_targets(val_df, 'Exited')
    
    numeric_cols, categorical_cols = identify_column_types(train_inputs)
    
    train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols, scaler_numeric)
    train_inputs, val_inputs, encoder = one_hot_encode_categorical_features(train_inputs, val_inputs, categorical_cols)
    
    input_cols = train_inputs.columns.tolist()

    result = {
        'train_X': train_inputs,
        'train_y': train_targets,
        'val_X': val_inputs,
        'val_y': val_targets,
        'scaler': scaler,
        'encoder': encoder
    }
    
    return result


def preprocess_new_data(new_data: pd.DataFrame, scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Preprocess new data using the trained scaler and encoder.
    
    Args:
        new_data (pd.DataFrame): The new data to preprocess.
        scaler (MinMaxScaler): The trained scaler.
        encoder (OneHotEncoder): The trained encoder.
        
    Returns:
        pd.DataFrame: The preprocessed new data.
    """
    # Drop columns that were dropped during training
    new_data = new_data.drop(columns=['CustomerId', 'Surname'], errors='ignore')

    #Identify numeric and categorical columns
    numeric_cols, categorical_cols = identify_column_types(new_data)

    # Scale numeric features
    if scaler is not None:
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    #scaler.fit(new_data[numeric_cols])
    #new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    
    # One-hot encode categorical features
    encoded_cols = encoder.get_feature_names_out(encoder.feature_names_in_)
    new_encoded = pd.DataFrame(encoder.transform(new_data[encoder.feature_names_in_]), columns=encoded_cols, index=new_data.index)
    new_data = new_data.drop(columns=encoder.feature_names_in_).join(new_encoded)
    
    return new_data

