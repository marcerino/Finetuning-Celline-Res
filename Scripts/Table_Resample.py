import os 
import polars as pl 


def atomicstats(path: str, Model : str = "Unkonwn",Samples :str = None, Run:str = None)-> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.select(["metric","value"]).transpose(column_names = "metric")
    df = pl.concat([pl.DataFrame({"Model":Model,"Samples": Samples, "Run":Run}),df], how = "horizontal" )
    return  df


def totaltable(path_to_dir: str)-> pl.DataFrame:
    def args_for_Model(filename:str) -> dict[str:str]:
        args = {"path":filename, "Model" :None,"Samples" :None,"Run" :None}
        if "RandomForest" in filename.split("/")[-1]:
            args["Model"] = "RandomForest"
        elif "supervised_vae" in filename:
            args["Model"] = "supervised_vae"

        args["Samples"] = filename.split("/")[-1].split(".")[0].split("_")[-2]
        args["Run"] = filename.split("/")[-1].split(".")[0].split("_")[-1]

        return args

    stats = [os.path.join(path_to_dir,i) for i in os.listdir(path_to_dir) if i.endswith(".stats.csv")]
    stats_and_args = [args_for_Model(statpth) for statpth in stats]
    print("Reading")
    
    dataframes = [atomicstats(argset["path"],argset["Model"],argset["Samples"],argset["Run"]) for argset in stats_and_args ]
    print("Concating")
    return pl.concat(dataframes,how = "vertical")
    
    
def totaltable_finetune(path_to_dir: str)-> pl.DataFrame:
    def args_for_Model(filename:str) -> dict[str]:
        args = {"path":filename, "Model" :None,"Samples" :None,"Run" :None}
        #if "RandomForest" in filename.split("/")[-1]:
            #args["Model"] = "RandomForest"
            #elif "supervised_vae" in filename:
        args["Model"] = "finetune_supervised_vae"
        #0_FinetuneSamples_resample_run_0.stats.csv
        args["Samples"] = filename.split("/")[-1].split(".")[0].split("_")[0]
        args["Run"] = filename.split("/")[-1].split(".")[0].split("_")[-1]

        return args

    allfiles_inpth =[]
    for root, dirs, files in os.walk(path_to_dir, topdown=False):
        for name in files:
                 if os.path.join(root,name).endswith(".stats.csv"):
                     allfiles_inpth.append(os.path.join(root,name))
    

             
    stats_and_args = [args_for_Model(statpth) for statpth in allfiles_inpth]
    print("Reading")
    
    dataframes = [atomicstats(argset["path"],argset["Model"],argset["Samples"],argset["Run"]) for argset in stats_and_args ]
    print("Concating")
    return pl.concat(dataframes,how = "vertical")



if __name__ == "__main__":
    headdir = os.path.abspath(os.pardir)
    resdir = os.path.join(headdir,"Models","Resample")
    plotdir = os.path.join(headdir,"Plots")
    finetuned = os.path.join(headdir,"size_filterd","Resampled")
    from_scratch_save = os.path.join(plotdir,"Resample_from_scratch_stats.csv")
    finetune_save = os.path.join(plotdir,"Resample_finetuned_stats.csv")
    
    tdf = totaltable(resdir)
    tdf.write_csv(from_scratch_save)

    
    
    tdf = totaltable_finetune(finetuned)
    tdf.write_csv(finetune_save)