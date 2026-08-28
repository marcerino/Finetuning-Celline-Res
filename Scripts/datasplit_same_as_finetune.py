import os
import polars as pl

def get_finetunesamples(pth_to_pred_labels_csv) -> list[str]:
    df = pl.read_csv(pth_to_pred_labels_csv)
    df = df.filter(pl.col("split")=="finetune").select("sample_id").unique()

    return df.to_series().to_list()


def split_dataset_by_finetune_samples(pth_to_pred_labels_csv :str,pth_to_dataset :str , outputdir :str):
    finetunelabels = get_finetunesamples(pth_to_pred_labels_csv)
    test_labels =  pl.read_csv(pth_to_pred_labels_csv).filter(pl.col("split")=="test").select("sample_id").unique().to_series().to_list()

    clin = pl.read_csv(os.path.join(pth_to_dataset,"clin.csv"))
    gex = pl.read_csv(os.path.join(pth_to_dataset,"gex.csv"))

    trclin = clin.filter(pl.col("ModelID").is_in(finetunelabels))
    trclin.write_csv(os.path.join(outputdir, "train","clin.csv"))

    teclin = clin.filter(pl.col("ModelID").is_in(test_labels))
    teclin.write_csv(os.path.join(outputdir, "test","clin.csv"))

    trgex = gex.select([gex.columns[0]]+finetunelabels)
    trgex.write_csv(os.path.join(outputdir, "train","gex.csv"))

    tegex = gex.select([gex.columns[0]]+test_labels)
    tegex.write_csv(os.path.join(outputdir, "test","gex.csv"))


    with open(os.path.join(outputdir, "test","gex.csv"), "r") as f:
       lines = f.readlines()
       if lines and lines[0].startswith('""'):
           lines[0] = lines[0][2:]
    with open(os.path.join(outputdir, "test","gex.csv"), "w") as f:
       f.writelines(lines)


    with open(os.path.join(outputdir, "train","gex.csv"), "r") as f:
       lines = f.readlines()
       if lines and lines[0].startswith('""'):
           lines[0] = lines[0][2:]
    with open(os.path.join(outputdir, "train","gex.csv"), "w") as f:
       f.writelines(lines)


if __name__ == "__main__":
    headdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(headdir, "Data")
    resultdir = os.path.join(headdir, "size_filterd", "geq_25_samples")

    pred_files = [file for file in os.listdir(resultdir) if file.endswith(".predicted_labels.csv")]

    outnames = [file.split(".")[0]+"_split" for file in pred_files] # ie. 75_FinetuneSamples_split


    pred_path = [os.path.join(resultdir,i) for i in pred_files ]
    originaldatapth = "/data/local/mgiller/atlas_tissue_representation/Data/more_than_25_samples"

    for i in range(len(pred_path)):
        os.makedirs(os.path.join(datadir,outnames[i],"test"),exist_ok = True)
        os.makedirs(os.path.join(datadir,outnames[i],"train"),exist_ok = True)
        split_dataset_by_finetune_samples(pred_path[i],originaldatapth,os.path.join(datadir,outnames[i]))
