import pandas as pd
from typing import Tuple


def train_test_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разбивает датафрейм на обучающую и тестовую выборки.

    Parameters:
    -----------
    df : pandas DataFrame
        Исходный датафрейм.
    test_size : float
        Размер тестовой выборки в долях от единицы.

    Returns:
    --------
    train_df : pandas DataFrame
        Обучающая выборка.
    test_df : pandas DataFrame
        Тестовая выборка.
    """
    n_test = int(len(df) * test_size)
    n_train = len(df) - n_test

    # Случайное перемешивание индексов датафрейма
    shuffled_indices = df.sample(frac=1, random_state=42).index.tolist()

    # Разделение на обучающую и тестовую выборки
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:]
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]

    return train_df, test_df
