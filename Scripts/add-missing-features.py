import polars as pl

def add_missing_features(df: pl.DataFrame, missing_features: list[str], id_col: str = "") -> pl.DataFrame:
    numeric_cols = [c for c in df.columns if c != id_col and df.schema[c].is_numeric()]

    # fast row-wise mean via numpy
    arr = df.select(numeric_cols).to_numpy()
    row_mean = arr.mean(axis=1)  # one value per row

    new_cols = [pl.Series(feat, row_mean) for feat in missing_features]
    return df.with_columns(new_cols)



def add_missing_features_as_zeros(df: pl.DataFrame, missing_features: list[str]) -> pl.DataFrame:
    # add each missing feature as a column of 0.0
    missing_df = pl.DataFrame({feat: [0.0] * df.height for feat in missing_features})
    return pl.concat([df, missing_df], how="horizontal")

if __name__ == "__main__":


    missing_features = ["TUG1", "METTL13"]
    path_to_csv = "/data/local/mgiller/atlas_tissue_representation/celline-data/gex-no-id.csv"

    df = pl.read_csv(path_to_csv)
    df = add_missing_features(df, missing_features)
    print(df.head())

    input("press enter to transpose")
    df = df.transpose(include_header=True )#header_name="gene", column_names=df.columns)
    print(df.tail())

    input("press enter to save")
    df.write_csv("/data/local/mgiller/atlas_tissue_representation/celline-data/gex_no_id_interpolateddata.csv")
