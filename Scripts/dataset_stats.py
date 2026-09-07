import os
from threading import local
import polars as pl
import tomlkit
import yaml

def usable_amount(pthtodir:str)-> pl.DataFrame:
    gex = pl.read_csv(os.path.join(pthtodir, "gex.csv"))
    clin = pl.read_csv(os.path.join(pthtodir, "clin.csv"))

    clean_clin = clin.filter(pl.nth(0).is_in(gex.columns))
    return clean_clin

def dict_from_clasifiyer(clin_df:pl.DataFrame, classification_feature:str):
    return dict(clin_df.group_by(classification_feature).len().sort(by=classification_feature).rows())

def summarize_predlable(pth:str):
    df = pl.read_csv(pth)
    localsummary = {}
    if "finetune" in df["split"]:
        localsummary["Nr_finetune"] = df.filter(pl.col("split") == "finetune").unique("sample_id").shape[0]
        localsummary["finetune_labels"] = dict(df.filter(pl.col("split") == "finetune").unique("sample_id").group_by("known_label").len().sort(by="known_label").rows())
    if "train" in df["split"]:
        localsummary["Nr_train"] = df.filter(pl.col("split") == "train").unique("sample_id").shape[0]
        localsummary["train_labels"] = dict(df.filter(pl.col("split") == "train").unique("sample_id").group_by("known_label").len().sort(by="known_label").rows())

    if "test" in df["split"]:
        localsummary["test_labels"] = dict(df.filter(pl.col("split") == "test").unique("sample_id").group_by("known_label").len().sort(by="known_label").rows())
        localsummary["Nr_test"] = df.filter(pl.col("split") == "test").unique("sample_id").shape[0]
        
    return localsummary

def create_summary(pthtodir:str, headdir:str,classification_feature:str):
    summary = {}
    pth = pthtodir
    if os.path.exists(os.path.join(pth, "clin.csv")):
        name = os.path.basename(pth)
        clin = usable_amount(pth)
        gex = pl.read_csv(os.path.join(pth, "gex.csv"))
        summary[name] = {"clin_path": os.path.join(pth, "clin.csv")}
        summary[name]["Amount_samples"] = clin.shape[0]
        summary[name]["Amount_Features"] = gex.shape[0]
        summary[name]["classification_feature"] = classification_feature
        summary[name][classification_feature] = dict_from_clasifiyer(clin, classification_feature)
    elif any([file.endswith("predicted_labels.csv") for file in os.listdir(pth)]):
        for file in os.listdir(pth):
            if file.endswith(".predicted_labels.csv"):
                name = os.path.basename(file).replace(".predicted_labels.csv", "")
                summary[name] = {}
                summary[name]["Name"] = name
                summary[name]["predicted_labels_path"] = os.path.join(pth, file)
                summary[name]["Summary"] = summarize_predlable(os.path.join(pth, file))
                

    yaml.dump(summary, open(os.path.join(pthtodir, "summary.yaml"), "w"))
    # with open(os.path.join(pthtodir, "summary.toml"), "w") as f:
    #     tomlkit.dump(summary, f)
        
if __name__ == "__main__":
    headdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(headdir, "Data")
    geq25dir =  os.path.join(datadir,"more_than_25_samples")
    create_summary(geq25dir, headdir, "uberon_tissue")
    #125_FinetuneSamples_resample_run_1.predicted_labels.csv
    create_summary(os.path.join(headdir, "size_filterd", "Resampled", "resample_1"), headdir, "uberon_tissue")

    summarylist = [os.path.join(headdir, "size_filterd", "Resampled", "resample_" + str(i)) for i in range(0, 10)]
    summarylist += [os.path.join(headdir, "Models", "Resample")]
    for summarydir in summarylist:
        create_summary(summarydir, headdir, "uberon_tissue")
    
