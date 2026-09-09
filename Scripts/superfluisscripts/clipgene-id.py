import polars as pl

def strip_geneid_from_attributes(filepth):
    df = pl.read_csv(filepth)
    df = df.rename(lambda column_name: column_name.split("(")[0].strip())

    print("###ATRIBUTES###")
    for i in df.columns:
        print(i)
    awns = input("Does this seeem Correct? y/n: ")
    if awns == "y":
              df.write_csv(filepth+"new.csv")    # worst thing i have written so far  but hey it is just this little operation )

if __name__ == "__main__":
              filepath = "/data/local/mgiller/atlas_tissue_representation/celline-data/old-gex.csv"
              strip_geneid_from_attributes(filepath)
