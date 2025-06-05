import polars as pl
from sklearn.feature_extraction import DictVectorizer


def fit_transform_dictvect(df: pl.DataFrame, dv: DictVectorizer = None):
    feature_mapping = df.to_dicts()
    dv = DictVectorizer(sparse=True)
    x_train = dv.fit_transform(feature_mapping)
    return x_train, dv


def apply_dictvect(df: pl.DataFrame, dv: DictVectorizer):
    feature_mapping = df.to_dicts()
    return dv.transform(feature_mapping)
