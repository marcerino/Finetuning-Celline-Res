from pathlib import Path

import h5py
import joblib
import polars as pl


def features_joblib(filepath: str | Path) -> list[str]:
    artifacts = joblib.load(filepath)
    if len(artifacts["feature_lists"]) == 1:
        return list(artifacts["feature_lists"][list(artifacts["feature_lists"])[0]])
    else:
        print("Multiple feature lists found. Please select one:")
        for i, features in enumerate(artifacts["feature_lists"]):
            print(f"{i + 1}: {features}")
    while True:
        choice_input = input("Which key describes the intended target best? (Enter number): ")
        choice_idx = int(choice_input) - 1
        if 0 <= choice_idx < len(artifacts["feature_lists"]):
            break
        print(f"Please enter a number between 1 and {len(artifacts['feature_lists'])}.")

    return artifacts["feature_lists"][choice_idx]

def features_csv(filepath: str | Path) -> list[str]:
    """Extract column names from a CSV file efficiently without loading all rows."""
    return pl.read_csv(filepath, n_rows=0).columns

def features_h5(filepath: str | Path) -> list[str]:
    """Interactively select an HDF5 dataset key and return its feature/dtype names."""
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())

        if not keys:
            raise ValueError(f"No keys found in HDF5 file: {filepath}")

        if "feature_names" in f.keys():
            return list(f["feature_names"].asstr())


        choice_idx = 0
        print(f"\nHDF5 file: {filepath}")
        for idx, key in enumerate(keys, 1):
            print(f"{idx}: {key}")
        while True:
            try:
                choice_input = input("Which key describes the intended target best? (Enter number): ")
                choice_idx = int(choice_input) - 1
                if 0 <= choice_idx < len(keys):
                    break
                print(f"Please enter a number between 1 and {len(keys)}.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

        selected_dataset = f[keys[choice_idx]].asstr()

        return(list(selected_dataset))

        """
        # Check if the dataset has named fields (structured array)
        if hasattr(selected_dataset.dtype, "names") and selected_dataset.dtype.names:
            return list(selected_dataset.dtype.names)
        else:
            # Fallback if it's a standard numeric array
            return [f"dim_{i}" for i in range(selected_dataset.shape[1])] if len(selected_dataset.shape) > 1 else []
        """
def getfeatures(filepath: str | Path) -> set[str]:
    """Determine file type and return its feature names."""
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return features_csv(path)
    elif suffix in [".h5", ".hdf5"]:
        return features_h5(path)
    elif suffix == ".joblib":
        return features_joblib(path)
    else:
        raise ValueError(f"Does not match supported filetypes (.csv, .h5): {filepath}")

def sharedfeatures(afilepth: str | Path, bfilepth: str | Path) -> set[str]:
    af = set(getfeatures(afilepth))
    #print("af:", af)
    bf = set(getfeatures(bfilepth))
    #print("bf:", bf)
    return af & bf

def difffeatures(afilepth: str | Path, bfilepth: str | Path) -> set[str]:
    af = set(getfeatures(afilepth))
    #print("af:", af)
    bf = set(getfeatures(bfilepth))
    #print("bf:", bf)
    return af - bf

def extrafeatures(afilepth: str | Path, bfilepth: str | Path) -> set[str]:
    af = set(getfeatures(afilepth))
    #print("af:", af)
    bf = set(getfeatures(bfilepth))
    #print("bf:", bf)
    #print(f"{len(bf) = }")
    return bf - af


if __name__ == "__main__":
    o = Path("/data/local/mgiller/atlas_tissue_representation/processed_scaled_411k_tissue_B_h5/train/gex.h5")
    jobl = Path("/home/mgiller/Atlasmodel/Atlas-uberon_tissue-SupervisedVAE/fullfeatures.artifacts.joblib")
    cel = Path("/data/local/mgiller/atlas_tissue_representation/celline-data/gex-no-id-nodecoration.csv")

    fpa = jobl
    fpb = cel


    sf = sharedfeatures(fpa, fpb)
    print(f"{fpa}\n{fpb} \nShared Features :", len(sf))
    df = difffeatures(fpa, fpb)
    print(f"{fpa}\n{fpb} \nDiff Features (in a not in b):", len(df))
    for diff in df:
        print(diff)
    ef = extrafeatures(fpa, fpb)
    print(f"{fpa}\n{fpb} \nExtra Features (in b not in a):", len(ef))
