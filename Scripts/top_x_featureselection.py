import polars as pl

def getcsv_vals(csv_path,original_col):
    df = pl.read_csv(csv_path)
    return set(df[original_col].unique())

def apply_tissue_mapping(target_df: pl.DataFrame, tissue_mapping: dict[str, str], target_col: str, original_col: str):
    target_df = target_df.with_columns(
        pl.col(target_col)
        .replace_strict(tissue_mapping, default=None)
        .alias(original_col)
    )
    return target_df

def select_top_x(df: pl.DataFrame, col: str, x: int) -> pl.DataFrame:
    top_values = (
        df.group_by(col)
        .len()
        .sort("len", descending=True)
        .head(x)
        .get_column(col)
    )
    print(top_values)
    input("labidasd")
    return df.filter(pl.col(col).is_in(top_values))

def select_features_with_more_than_x_samples(df: pl.DataFrame, col: str, x: int) -> pl.DataFrame:
    # Filter rows where the value in col appears more than x times
    df_filtered = df.filter(pl.col(col).len().over(col) >= x)

    print(df_filtered[col].unique().to_list())

    #print(ple.head())
    return df_filtered

def select_features_with_leq_than_x_samples(df: pl.DataFrame, col: str, x: int) -> pl.DataFrame:
    # Filter rows where the value in col appears more than x times
    df_filtered = df.filter((pl.col(col).len().over(col) <= x) & (pl.col(col).len().over(col) >= 5))

    print(df_filtered[col].unique().to_list())

    #print(ple.head())
    return df_filtered

if __name__ == "__main__":
    comparpth = "/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata/clin-Primdiseaseoncomapping-revised.csv"
    savleloc = "/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata/"
    original_col = "uberon_tissue"
    target_col = "uberon_tissue"


    for i in [15,25,35]:
        c =  pl.read_csv(comparpth)
        c = select_features_with_more_than_x_samples(c, original_col, i)
        c = c.select([pl.col("ModelID"),pl.col(original_col)]).filter(pl.col(original_col).is_not_null())
        print(c.columns)
        print(c.head())

        c.write_csv(savleloc + f"clin-more_than_{i}_samples-revised.csv")

    c =  pl.read_csv(comparpth)

    c = select_features_with_leq_than_x_samples(c, original_col, 60)
    c = c.select([pl.col("ModelID"),pl.col(original_col)]).filter(pl.col(original_col).is_not_null())
    print(c.columns)
    print(c.head())

    c.write_csv(savleloc + f"clin-leq_than_60_samples-revised.csv")
