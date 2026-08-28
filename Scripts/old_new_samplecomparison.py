import polars as pl

def comparecollums(dfo,dfn):
    sdfn = set(dfn.columns)
    sdfo = set(dfo[""].to_list())
    #print(sdfo)
    return sdfo - sdfn

def in_clin_not_gex(dfc,gdf,samplecol_col):
    return set(dfc[samplecol_col].to_list()) - set(gdf.columns)

if __name__ == "__main__":
    old_gex="/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata/old-gex.csv"
    new_gex = "/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata/gex-new.csv"
    clinpth = "/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata/clin-Primdiseaseoncomapping-revised.csv"

    dfc = pl.read_csv(clinpth)
    dfn = pl.read_csv(new_gex)
    dfo = pl.read_csv(old_gex)
    col = "ModelID"
    print(f"{comparecollums(dfo,dfn) = }")
    print(f"{in_clin_not_gex(dfc,dfn,col) = }")
