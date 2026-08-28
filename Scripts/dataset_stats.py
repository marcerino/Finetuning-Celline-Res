import os
import polars as pl

def usable_amount(pthtodir:str)-> pl.DataFrame:
    gex = pl.read_csv(os.path.join(pthtodir, "gex.csv"))
    clin = pl.read_csv(os.path.join(pthtodir, "clin.csv"))

    clean_clin = clin.

if __name__ == "__main__":
    headdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(headdir, "Data")
    geq25dir =  os.path.join(datadir,"more_than_25_samples")
