import pickle as pk

import polars as pl


def read_dataset(filename, mode="normal"):
    if mode == "normal":
        return pl.read_parquet(filename)
    elif mode == "lazy":
        return pl.scan_parquet(filename)
    else:
        raise ValueError("Choose `mode` between 'lazy' or 'normal'")


def get_num_rows(df):
    return df.select(pl.len()).collect().item()


def prepare_dataset(df):
    df = df.with_columns(
        duration=(
            (
                pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")
                # pl.col("lpep_dropoff_datetime") - pl.col("lpep_pickup_datetime")
            ).dt.total_seconds()
            / 60
        ),
        pu=pl.col("PULocationID").cast(str),
        do=pl.col("DOLocationID").cast(str),
        # pu_do=pl.concat_str(
        #     [  # pl.concat_str will cast to str automatically
        #         pl.col("PULocationID"),  # .cast(str),
        #         pl.col("DOLocationID"),  # cast(str),
        #     ],
        #     separator="_",
        # ),
    )

    df = df.filter(pl.col("duration").is_between(1, 60))

    categorical = ("pu", "do")  # ("pu_do",)
    numerical = ("trip_distance",)
    target = "duration"

    df_trainset = df.select(
        pl.col(categorical),
        # pl.col(numerical),
    ).collect()  # to_dicts()

    df_target = df.select(pl.col(target)).collect()

    return df_trainset, df_target


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pk.dump(obj, f_out)


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pk.load(f_in)
