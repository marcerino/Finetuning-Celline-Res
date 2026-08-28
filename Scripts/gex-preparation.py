import polars as pl
import joblib
import os
def strip_geneid_from_attributes(df)-> pl.DataFrame:
    df = df.rename(lambda column_name: column_name.split("(")[0].strip())
    return df


def features_joblib(filepath: str ) -> list[str]:
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

def add_missing_features(df: pl.DataFrame, missing_features: list[str], id_col: str = "") -> pl.DataFrame:
    numeric_cols = [c for c in df.columns if c != id_col]

    row_mean = df.select(numeric_cols).mean_horizontal()

    new_cols = [pl.Series(feat, row_mean) for feat in missing_features]
    return df.with_columns(new_cols)

def get_missing_features(df : pl.DataFrame, featurelist) -> list[str]:
    return list(set(featurelist) - set(df.columns) -set(""))



def add_missing_features_as_zeros(df: pl.DataFrame, missing_features: list[str]) -> pl.DataFrame:
    # add each missing feature as a column of 0.0
    missing_df = pl.DataFrame({feat: [0.0] * df.height for feat in missing_features})
    return pl.concat([df, missing_df], how="horizontal")


if __name__ == "__main__":
    headdir = os.path.dirname(os.getcwd())
    filepath = os.path.join(headdir,"Data","onco-cellinedata","old-gex.csv")
    pth_to_joblib = os.path.join(headdir,"Models","Atlas-uberon_tissue-SupervisedVAE","fullfeatures.artifacts.joblib")
    savepath = os.path.join(headdir,"Data","onco-cellinedata","gex-cleaned.csv")
    df = pl.read_csv(filepath)
    df = strip_geneid_from_attributes(df)

    missing_features = get_missing_features(df,features_joblib(pth_to_joblib))
    #input(f"{missing_features = }\nPress Ctl+C to abort")
    print("Calculating Mean")
    df = add_missing_features(df,missing_features)
    print(df.head())
    #input(df.tail())
    print("transposing")
    df = df.transpose(include_header=True, header_name="", column_names=df[:, 0])[1:] #[1:] because the transposing also inserts the former ID column as a row, now excluded
    print(df.head())
    input(df.tail())
    df.write_csv(savepath)

    with open(savepath, "r") as f:
       lines = f.readlines()

    if lines and lines[0].startswith('""'):
       lines[0] = lines[0][2:]

    with open(savepath, "w") as f:
       f.writelines(lines)
