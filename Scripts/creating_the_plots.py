from this import s

import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import polars as pl
import numpy as np
import re

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

def create_confusion_metrics(pth_to_csv: str,save_loc:str,Title : str,figsize = (12,10)):
    plt.clf()
    df = pd.read_csv(pth_to_csv)

    df = df.groupby("sample_id",sort=False, as_index=False).first()
    labels1 = df["known_label"].tolist()
    labels2 = df["predicted_label"].tolist()



    ct = pd.crosstab(pd.Series(labels1, name='known_label'), pd.Series(labels2, name='predicted_label'))
    # Normalize the cross-tabulation matrix column-wise
    ct_normalized = ct.div(ct.sum(axis=1), axis=0)

    # Plot the heatmap
    plt.figure(figsize = figsize)
    sns.heatmap(ct_normalized, annot=True,cmap='viridis', linewidths=.5)# col_cluster=False)
    plt.title(Title)
    plt.savefig(save_loc)

    plt.clf()
    plt.close()

def confusion_matrices_in_dir(path_to_dir,pth_outdir,Title):
    metrics = [file for file in os.listdir(path_to_dir) if  file.endswith(".predicted_labels.csv")]

    for i in metrics:
        figtitle = Title +"_"+ i.split(".")[0]
        create_confusion_metrics(os.path.join(path_to_dir,i),os.path.join(pth_outdir,figtitle+".png"),figtitle)


def extract_confusion_metrics(path_to_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv)
    df = df.groupby("sample_id", sort=False, as_index=False).first()
    df = df[["known_label", "predicted_label"]]

    ct = pd.crosstab(
        pd.Series(df['known_label'], name='Known Label'),
        pd.Series(df['predicted_label'], name='Predicted Label')
    )
    # Normalize the cross-tabulation matrix column-wise
    ct_normalized = ct.div(ct.sum(axis=0), axis=1)
    ct_normalized = ct.div(ct.sum(axis=1), axis=0)
    known_label = df["known_label"].unique().tolist()
    predicted_label = df["predicted_label"].unique().tolist()

    true_res = []
    for label in known_label:
        if label in predicted_label:
            true_res.append([label,ct_normalized.loc[label,label]])
        else:
            true_res.append([label,0])

    score_name = os.path.basename(path_to_csv).split(".")[0]
    return pd.DataFrame(true_res,columns = ["label",score_name])



def create_dataframe_over_confusionmetrics(path_to_dir):
    dfs = []
    scorenames = [file.split(".")[0] for file in os.listdir(path_to_dir) if  file.endswith(".predicted_labels.csv")]
    for filename in os.listdir(path_to_dir):
        if filename.endswith(".predicted_labels.csv"):
            file_path = os.path.join(path_to_dir, filename)
            df = extract_confusion_metrics(file_path)
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    #print(combined.head(29))
    result = combined.groupby("label", as_index=False).first().fillna(0)

    #print(result.head())
    return result


def lollipop_plot(df: pd.DataFrame, save_loc: str, Title="Label Concordance"):
    descriptors = list(df.columns.values)
    ordered_df = df.sort_values(by=descriptors[len(descriptors) - 1])
    my_range = range(1, len(df.index) + 1)

    palette = cm.viridis(np.linspace(0, 1, len(descriptors)))
    hex_colors = [mcolors.to_hex(c) for c in palette]
    category_color = dict(enumerate(hex_colors))  # {0: '#...', 1: '#...', ...}

    for i in range(1, len(descriptors) - 1):
        plt.hlines(
            y=my_range,
            xmin=ordered_df[descriptors[i]],
            xmax=ordered_df[descriptors[i + 1]],
            color='grey', alpha=0.4, zorder=1
        )
    for i in range(1, len(descriptors)):
        plt.scatter(
            ordered_df[descriptors[i]], my_range,
            color=category_color[i], alpha=1, label=descriptors[i]
        )

    plt.legend()
    plt.yticks(my_range, ordered_df['label'])
    plt.title(Title, loc='left')
    plt.xlabel('Label Concordance')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig(save_loc)
    plt.clf()
    plt.close()
    


def score_visualizer(path_to_dir:str,savedir:str):
    plt.clf()

    ### Reading the Files
    stats = [os.path.join(path_to_dir,f) for f in os.listdir(path_to_dir) if f.endswith("stats.csv") and not f.endswith("baseline.stats.csv")]

    frames = []
    for i in stats:
        sf = pl.read_csv(i)
        name = i.split("/")[-1].split(".")[0]
        sf = sf.with_columns(pl.lit(name).alias("name"))
        frames.append(sf)
        #print(sf.head())

    combined = pl.concat(frames)

    results = (
        combined.filter(pl.col("metric").is_in(["balanced_acc", "f1_score", "kappa"]))
        .pivot(
            values="value",
            index=["name", "method", "var", "variable_type"],
            on="metric",
        )
        .select(["name", "method", "var", "balanced_acc", "f1_score", "kappa"])
    )

    ## Creating the plot
    df = results
    df = df.with_columns(
        pl.col('name').str.extract(r'_(\d+)$', 1).cast(pl.Int64).alias('sort_key')
    ).sort('sort_key')

    metrics = ['balanced_acc', 'f1_score', 'kappa']

    sample_names = df['name'].unique(maintain_order=True).to_list()

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, name in enumerate(sample_names):
      row = df.filter(pl.col('name') == name)
      values = [row[m][0] for m in metrics]

      offset = x + (i * width) - width
      rects = ax.bar(offset, values, width, label=name)
      ax.bar_label(rects, fmt='%.3f', padding=3, fontsize=9)

    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Fine-Tuning Sample Size')
    ax.set_xticks(x)
    ax.set_xticklabels(['Balanced Acc', 'F1 Score', 'Kappa'])
    ax.set_ylim(0, 0.9)
    ax.legend(title='# of Training Samples', loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(savedir)
    plt.clf()
    plt.close()
    

def basleine_performance_visualyser(path_to_dir :str, saveloc :str , Title: str = "Baseline Performance"):
    files = [f for f in os.listdir(path_to_dir) if f.endswith("baseline.stats.csv")]
    metrics = ["balanced_acc", "f1_score", "kappa", "average_auroc", "average_aupr"]

    dfs = []
    for f in files:
        label = f.split(".")[0]
        d = pl.read_csv(os.path.join(path_to_dir, f)).with_columns(
            pl.lit(label).alias("source_file")
        )
        dfs.append(d)
    df = pl.concat(dfs)

    sort_df = df.select(["source_file"]).unique().with_columns(
        pl.col("source_file").str.extract(r"(\d+)", 1).cast(pl.Int64).alias("sort_key")
    )
    sort_df = sort_df.sort(["sort_key", "source_file"])
    source_order = sort_df["source_file"].to_list()

    methods = df["method"].unique(maintain_order=True).to_list()

    n_sources = len(source_order)
    width = 0.8 / n_sources
    x = np.arange(len(metrics))

    for method in methods:
        method_df = df.filter(pl.col("method") == method)

        fig, ax = plt.subplots(figsize=(9, 5.5))

        for i, source in enumerate(source_order):
            sub = method_df.filter(pl.col("source_file") == source)
            values = []
            for m in metrics:
                v = sub.filter(pl.col("metric") == m)["value"].to_list()
                values.append(v[0] if v else 0.0)

            offset = x + (i - (n_sources - 1) / 2) * width
            rects = ax.bar(offset, values, width, label=source)
            ax.bar_label(rects, fmt="%.3f", padding=3, fontsize=8, rotation=90)

        ax.set_ylabel("Score")
        ax.set_title(f"{Title}: {method}")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.0)
        ax.legend(title="Source", loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        safe_method = re.sub(r"[^\w\-.]", "_", method)
        local_save = os.path.join(saveloc, f"{safe_method}_baseline_comparison.png")
        plt.savefig(local_save)
        plt.close(fig)

def plotpermetric(dict_to_path_to_dir: list[str], saveloc: str, Title: str = "Performance"):
    def sortfunc(x):
        if "from_scratch" in x and "supervised_vae" in x:
            return 5001 + int(x.split("/")[-1].split(".")[0].split("_")[4])
        elif "from_scratch" in x and "RandomForest" in x:
            return 5000 + int(x.split("/")[-1].split(".")[0].split("_")[3])
        elif "geq" in x:
            return int(x.split("/")[-1].split(".")[0].split("_")[0])
        else:
            return 500

    def categorize(x):
        if "supervised_vae" in x:
            return "supervised_vae"
        elif "RandomForest" in x:
            return "RandomForest"
        elif "geq" in x:
            return "geq"
        else:
            return "other"

    # internal key -> display label. Edit freely without touching detection logic.
    CATEGORY_LABELS = {
        "geq": "Fine-Tune",
        "RandomForest": "Random Forest",
        "supervised_vae": "Supervised VAE",
        "other": "Other",
    }

    categories = list(CATEGORY_LABELS.keys())  # fixed order, controls color assignment
    palette = cm.viridis(np.linspace(0, 1, len(categories)))
    hex_colors = [mcolors.to_hex(c) for c in palette]
    category_color = dict(zip(categories, hex_colors))

    stat_files = []
    for d in dict_to_path_to_dir:
        stat_files += [
            os.path.join(d, i) for i in os.listdir(d)
            if i.endswith("stats.csv") and not i.endswith("baseline.stats.csv")
        ]
    stat_files.sort(key=sortfunc)

    file_to_category = {f: categorize(f) for f in stat_files}

    dfs = [
        pl.read_csv(file).with_columns(
            pl.lit(file.split("/")[-1].split(".")[0]).alias("title"),
            pl.lit(file_to_category[file]).alias("category"),
        )
        for file in stat_files
    ]
    concats = pl.concat(dfs, how="vertical")
    print(concats.head(20))
    print(concats["title"].unique().sort())

    metrics = ['balanced_acc', 'f1_score', 'kappa']
    for metric in metrics:
        metric_df = concats.filter(pl.col("metric") == metric)

        titles_ordered = metric_df["title"].to_list()
        cats_ordered = metric_df["category"].to_list()
        bar_colors = [category_color[c] for c in cats_ordered]

        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.bar(titles_ordered, metric_df["value"].to_list(), width=0.9, color=bar_colors)
        plt.xticks(range(len(titles_ordered)), titles_ordered, rotation=90)
        plt.title(f"{Title}: {metric}")

        # legend uses CATEGORY_LABELS for display text, only for categories present
        present = [c for c in categories if c in set(cats_ordered)]
        handles = [plt.Rectangle((0, 0), 1, 1, color=category_color[c]) for c in present]
        labels = [CATEGORY_LABELS[c] for c in present]
        plt.legend(handles, labels, title="Category")

        plt.tight_layout()
        plt.savefig(os.path.join(saveloc, Title + metric + ".png"), bbox_inches='tight')
        plt.clf()
        plt.close()
        
def plotpermetric_box(stat_files: list[str], saveloc: str, Title: str = "Performance"):
    def categorize(model: str) -> str:
        if "finetune" in model:
            return "geq"
        elif "RandomForest" in model:
            return "RandomForest"
        elif "supervised_vae" in model:
            return "supervised_vae"
        else:
            return "other"

    CATEGORY_LABELS = {
        "geq": "Finetune",
        "RandomForest": "RandomForest",
        "supervised_vae": "supervised_vae",
        "other": "Other",
    }
    CATEGORY_SUBRANK = {"geq": 0, "supervised_vae": 0, "RandomForest": 1, "other": 2}
    CATEGORY_BLOCK = {"geq": 0, "supervised_vae": 1, "RandomForest": 1, "other": 2}

    categories = list(CATEGORY_LABELS.keys())
    palette = cm.viridis(np.linspace(0, 1, len(categories)))
    hex_colors = [mcolors.to_hex(c) for c in palette]
    category_color = dict(zip(categories, hex_colors))

    stat_files = [
        f for f in stat_files
        if f.endswith("stats.csv") and not f.endswith("baseline.stats.csv")
    ]

    schema_overrides = {"average_auroc": pl.Float64, "average_aupr": pl.Float64}

    dfs = [
        pl.read_csv(file, schema_overrides=schema_overrides)
        .with_columns(pl.col("Model").map_elements(categorize, return_dtype=pl.Utf8).alias("category"))
        .with_columns(
            (pl.col("category").replace(CATEGORY_LABELS) + "_" + pl.col("Samples").cast(pl.Utf8)).alias("title")
        )
        for file in stat_files
    ]
    concats = pl.concat(dfs, how="vertical")

    metrics = ['balanced_acc', 'f1_score', 'kappa']
    id_vars = [c for c in concats.columns if c not in metrics]
    long_df = concats.unpivot(on=metrics, index=id_vars, variable_name="metric", value_name="value")

    for metric in metrics:
        metric_df = long_df.filter(pl.col("metric") == metric)

        cat_per_title = (
            metric_df.select(["title", "category", "Samples"])
            .unique(subset=["title"], keep="first")
            .with_columns(
                pl.col("category").replace(CATEGORY_BLOCK).cast(pl.Int64).alias("_block"),
                pl.col("category").replace(CATEGORY_SUBRANK).cast(pl.Int64).alias("_subrank"),
            )
            .sort(["_block", "Samples", "_subrank"])
        )
        titles_ordered = cat_per_title["title"].to_list()
        cats_ordered = cat_per_title["category"].to_list()
        bar_colors = [category_color[c] for c in cats_ordered]

        values_per_title = [
            metric_df.filter(pl.col("title") == t)["value"].to_list()
            for t in titles_ordered
        ]
        means_per_title = [float(np.mean(v)) for v in values_per_title]

        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 6))

        bp = ax.boxplot(
            values_per_title,
            positions=range(len(titles_ordered)),
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(bp['boxes'], bar_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color('black')

        rng = np.random.default_rng(0)
        for i, (vals, color) in enumerate(zip(values_per_title, bar_colors)):
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                [i + j for j in jitter], vals,
                color=color, edgecolor='black', linewidth=0.5, zorder=3, s=25,
            )

        # --- mean label above each box ---
        all_vals = [v for vals in values_per_title for v in vals]
        y_range = (max(all_vals) - min(all_vals)) if all_vals else 1.0
        text_offset = 0.03 * y_range if y_range > 0 else 0.01
        for i, (vals, mean_val) in enumerate(zip(values_per_title, means_per_title)):
            top = max(vals) if vals else mean_val
            ax.text(
                i, top + text_offset, f"{mean_val:.3f}",
                ha='center', va='bottom', fontsize=7, rotation=90,
            )

        # --- crossover lines: first Samples size where mean(supervised_vae) >= mean(geq) ---
        means_by_cat_size = (
            metric_df.filter(pl.col("category").is_in(["geq", "supervised_vae"]))
            .group_by(["category", "Samples"])
            .agg(pl.col("value").mean().alias("mean_value"))
        )
        finetune_means = dict(
            zip(
                means_by_cat_size.filter(pl.col("category") == "geq")["Samples"].to_list(),
                means_by_cat_size.filter(pl.col("category") == "geq")["mean_value"].to_list(),
            )
        )
        svae_means = dict(
            zip(
                means_by_cat_size.filter(pl.col("category") == "supervised_vae")["Samples"].to_list(),
                means_by_cat_size.filter(pl.col("category") == "supervised_vae")["mean_value"].to_list(),
            )
        )
        common_sizes = sorted(set(finetune_means) & set(svae_means))
        crossover_size = next(
            (s for s in common_sizes if svae_means[s] >= finetune_means[s]), None
        )
        if crossover_size is not None:
            ft_val = finetune_means[crossover_size]
            svae_val = svae_means[crossover_size]
            ax.axhline(
                ft_val, color=category_color["geq"], alpha=0.35,
                linestyle='--', linewidth=1.5,
                label=f"Finetune mean @ n={crossover_size} ({ft_val:.3f})",
            )
            ax.axhline(
                svae_val, color=category_color["supervised_vae"], alpha=0.35,
                linestyle='--', linewidth=1.5,
                label=f"supervised_vae mean @ n={crossover_size} ({svae_val:.3f})",
            )

        ax.set_xticks(range(len(titles_ordered)))
        ax.set_xticklabels(titles_ordered, rotation=90)
        ax.set_title(f"{Title}: {metric}")
        ax.margins(y=0.12)  # headroom so mean labels don't get clipped

        present = [c for c in categories if c in set(cats_ordered)]
        handles = [plt.Rectangle((0, 0), 1, 1, color=category_color[c]) for c in present]
        labels = [CATEGORY_LABELS[c] for c in present]
        if crossover_size is not None:
            line_handles, line_labels = ax.get_legend_handles_labels()
            handles += line_handles
            labels += line_labels
        ax.legend(handles, labels, title="Category", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(saveloc, Title + metric + "_box.png"), bbox_inches='tight')
        plt.clf()
        
if __name__ == "__main__":
    headdir = os.path.dirname(os.getcwd())
    datadir = os.path.join(headdir, "Data")
    resultdir = os.path.join(headdir, "size_filterd")
    plotdir = os.path.join(headdir, "Plots")
    os.makedirs(plotdir, exist_ok=True)
    save_loc = os.path.join(plotdir,"lollipopplot_25_and_more_samples_Finetuning.png")

    dirlowDirect = os.path.join(headdir, "Models","low_data_split","DirectPred")
    dirlowsupervae = os.path.join(headdir, "Models","low_data_split","supervised_vae")

    dir25 = os.path.join(resultdir, "geq_25_samples")
    dirtop9 = os.path.join(headdir, "oncotop9-finetune")
    dir25step10 = os.path.join(resultdir, "geq_25_samples")

    plotpermetric_box(["/data/local/mgiller/atlas_tissue_representation/Plots/Resample_finetuned_stats.csv",
        "/data/local/mgiller/atlas_tissue_representation/Plots/Resample_from_scratch_stats.csv"],
        plotdir,
        "Performance of Finetuned Models geq 25")

    plotpermetric(["/data/local/mgiller/atlas_tissue_representation/size_filterd/geq_25_samples","/data/local/mgiller/atlas_tissue_representation/Models/like_finetune_split"], plotdir, "Performance of Models geq 25")

    #25 and more flexy
    df = create_dataframe_over_confusionmetrics(dir25)
    lollipop_plot(df,save_loc)

    #onco top 9 Tissues and more flexy
    df = create_dataframe_over_confusionmetrics(dirtop9)
    lollipop_plot(df,os.path.join(plotdir,"top_9_tissues.png"))

    #Low Sample Training
    df = create_dataframe_over_confusionmetrics(dirlowDirect)
    lollipop_plot(df,os.path.join(plotdir,"Low_Data_Direct_tissues.png"))
    df = create_dataframe_over_confusionmetrics(dirlowsupervae)
    lollipop_plot(df,os.path.join(plotdir,"Low_Data_supervae_tissues.png"))

    #confusion Metrics
    os.makedirs(os.path.join(plotdir,"geq_25_confusion_table"), exist_ok=True)
    confusion_matrices_in_dir(dir25,os.path.join(plotdir,"geq_25_confusion_table"),"geq25confusionmatrices_with_Finetunesamples")
    
    #Lowsampleplots
    os.makedirs(os.path.join(plotdir,"low_sample_table"), exist_ok=True)
    confusion_matrices_in_dir(dirlowDirect,os.path.join(plotdir,"low_sample_table"),"low_train_sample_DirectPred_cofusion_metrics")
    confusion_matrices_in_dir(dirlowsupervae,os.path.join(plotdir,"low_sample_table"),"low_train_sample_supervised_vae_cofusion_metrics")

  
  
    """
    #Basline_lowsample
    basleine_performance_visualyser(os.path.join(dirlowDirect),os.path.join(plotdir,"low_sample_table"),"Baseline Performace 108 Samples")
    basleine_performance_visualyser(os.path.join(dirlowDirect),os.path.join(plotdir,"low_sample_table"),"Baseline Performace 50 Samples")

  
    #Kappa Scores  min 25
    score_visualizer(dir25,os.path.join(plotdir,"min_25_samples_scores.png"))

    score_visualizer(dir25step10,os.path.join(plotdir,"geq_25_samples_scores.png"))


    #kapa score top 9
    score_visualizer(dirtop9,os.path.join(plotdir,"top_9_stats.png"))

    #Kappascore Lowsample gex
    score_visualizer(dirlowDirect,os.path.join(plotdir,"low_dir_Direct_stats.png"))
    score_visualizer(dirlowsupervae,os.path.join(plotdir,"low_dir_supervae_stats.png"))


    """
