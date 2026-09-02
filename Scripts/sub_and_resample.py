import polars as pl

def getcsv_vals(csv_path,original_col):
    df = pl.read_csv(csv_path)
    return set(df[original_col].unique())

def select_features_with_more_than_x_samples(df: pl.DataFrame, col: str, x: int) -> pl.DataFrame:
    # Filter rows where the value in col appears more than x times
    df_filtered = df.filter(pl.col(col).len().over(col) >= x)

    print(df_filtered[col].unique().to_list())

    #print(ple.head())
    return df_filtered

import os
import polars as pl


def create_samples_min_1(
    df: pl.DataFrame,
    col: str,
    ntotal: int,
    set_samples: list[str] | None = None,
    seed: int = 0,
) -> list[list[str]]:
    if df.shape[0] < ntotal:
        raise ValueError("Not enough samples in subsampling")
    index_col = df.columns[0]

    if set_samples is None:
        set_samples_df = (
            df.group_by(col, maintain_order=True)
            .map_groups(lambda g: g.sample(n=1, seed=seed))
            .select(df.columns)
        )
    else:
        set_samples_df = df.filter(pl.col(index_col).is_in(set_samples))

    n_remaining = ntotal - set_samples_df.shape[0]
    if n_remaining < 0:
        raise ValueError("set_samples already has more rows than ntotal")

    remaining_pool = df.filter(~df[index_col].is_in(set_samples_df[index_col].implode()))
    if n_remaining > remaining_pool.shape[0]:
        raise ValueError("Not enough remaining samples to reach ntotal")

    rest_df = remaining_pool.sample(n=n_remaining, seed=seed).select(df.columns)

    full_df = pl.concat([set_samples_df, rest_df])
    return [full_df[index_col].to_list(), set_samples_df[index_col].to_list()]


if __name__ == "__main__":
    clinpth = "/data/local/mgiller/atlas_tissue_representation/Data/more_than_25_samples/clin.csv"
    outdir = "/data/local/mgiller/atlas_tissue_representation/Data/more_than_25_samples/subsampling"
    c = pl.read_csv(clinpth)
    os.makedirs(outdir,exist_ok=True)
    for j in range(0, 10):
        hadsamples = None
        
        for i in [50, 75, 100, 125, 150, 175, 200]:
            samplelist, hadsamples = create_samples_min_1(c, "uberon_tissue", i, hadsamples, j)
            outpath = os.path.join(outdir, f"Samplelist_{i}_samples_{j}_version.txt")
            with open(outpath, "w") as f:
                for line in samplelist:
                    f.write(f"{line}\n")