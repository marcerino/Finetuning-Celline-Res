import polars as pl
import h5py 
import re

def features_csv(filepath): return (pl.read_csv(filepath).attrs)
def features_h5(filepath): return (h5py.File(filepath).dtype.names)
def getfeatures(filepath):
        if  re.search(".csv$",filepath): return features_csv(filepath)
        elif  re.search(".h5$", filepath): 
            with h5py.File(filepath) as f:
                it = 1
                for i in f.keys():
                    print(it,i)
                    it = it + 1
                choice = input("Which key describes the intended target best? [TODO using number]: " )
                return f[choice].dtype.names

            
              
        else:  print("Does not match supported filetypes:", filepath)

def sharedfeatures(afilepth,bfilepth):
    af = set(getfeatures(afilepth))
    bf = set(getfeatures(bfilepth))
    return (af & bf)













if __name__ == "__main__":
    fpa = "/data/local/mgiller/atlas_tissue_representation/processed_scaled_411k_tissue_B_h5/train/gex.h5"
    fpb = "/data/local/mgiller/atlas_tissue_representation/celline-data/gex.csv"
    sf = sharedfeatures(fpa,fpb)
    print(sf)
