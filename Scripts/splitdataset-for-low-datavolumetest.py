import polars as pl
import os

def adjust_modality(df: pl.DataFrame, ncdf: pl.DataFrame) -> list[pl.DataFrame]:
    train_samples = ncdf[ncdf.columns[0]].to_list()
    #print(f"train_samples: {set(train_samples)}")
    train_df = df.select([df.columns[0]] + train_samples)
    test_df = df.select(pl.exclude(train_samples))
    return [train_df, test_df]


def evenly_split(df: pl.DataFrame, n: int, col: str, seed: int = 111) -> list[pl.DataFrame]:
    index_col = df.columns[0]
    n_values_per = (
        df.group_by(col, maintain_order=True)
        .map_groups(lambda g: g.sample(n=min(n, g.height), seed=seed))
        .select(df.columns)
    )
    print(n_values_per.head(30))
    #input("asas")
    rest = df.filter(~df[index_col].is_in(n_values_per[index_col].implode()))
    #print(f"n_values_per: {n_values_per.shape[0]}, rest: {rest.shape[0]}")
    return [n_values_per, rest]

if __name__ == "__main__":
    datadir = "/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata"
    outdir = "/data/local/mgiller/atlas_tissue_representation/Data/low-data-split"
    cdf = pl.read_csv(os.path.join(datadir, "clin.csv"))
    gdf = pl.read_csv(os.path.join(datadir, "gex.csv"))

    train_ncdf, test_ncdf = evenly_split(cdf, 6, "uberon_tissue")
    train_df, test_df = adjust_modality(gdf, train_ncdf)

    print(train_df.head())
#    print(test_df.head())

    train_df.write_csv(os.path.join(outdir, "train", "gex.csv"))
    test_df.write_csv(os.path.join(outdir, "test", "gex.csv"))
    train_ncdf.write_csv(os.path.join(outdir, "train", "clin.csv"))
    test_ncdf.write_csv(os.path.join(outdir, "test", "clin.csv"))
