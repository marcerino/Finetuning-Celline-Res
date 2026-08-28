import polars as pl

def getcsv_vals(csv_path : str, original_col: str) -> set[str]:
    df = pl.read_csv(csv_path)
    return set(df[original_col].unique())

def apply_tissue_mapping(target_df : pl.DataFrame, tissue_mapping: dict[str, str | None], target_col: str, original_col: str):
    target_df = target_df.with_columns(
        pl.col(target_col)
        .replace_strict(tissue_mapping, default=None)
        .alias(original_col)
    )
    return target_df

def samples_in_modality(dfclin,dfmod,sampleIDcol):
    return  set(dfclin[sampleIDcol].to_list()) & set(dfmod.columns)

if __name__ == "__main__":
    #orignpth = ""
    comparpth = "/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata/clin-old.csv"
    savleloc ="/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata/clin-Primdiseaseoncomapping-revised.csv"
    original_col = "uberon_tissue"
    target_col = "OncotreePrimaryDisease"
    gexpth = "/data/local/mgiller/atlas_tissue_representation/Data/onco-cellinedata/gex-new.csv"
    samplecol = "ModelID"
    #origin = getcsv_vals(orignpth, original_col)
    #compare = getcsv_vals(comparpth, target_col)
    #print(f"origin: {origin}\n\n\ncompare: {compare}\n\n\n\n{compare - origin = }\n\n\n\n{origin - compare = }\n\n\n{origin & compare = }")

    #Tissue mapping created via gemini pro 3.1
    tissue_mapping_gemini = {
        # --- Exact Matches (origin & compare) ---
        'eye': 'eye',
        'soft_tissue': 'soft_tissue',
        'stomach': 'stomach',
        'pleura': 'pleura',
        'kidney': 'kidney',
        'liver': 'liver',
        'biliary_tract': 'biliary_tract',
        'cervix': 'cervix',
        'small_intestine': 'small_intestine',
        'lung': 'lung',
        'ovary': 'ovary',
        'bone_marrow': 'bone_marrow',
        'spleen': 'spleen',
        'fibroblast': 'fibroblast',
        'pancreas': 'pancreas',
        'prostate': 'prostate',
        'thyroid': 'thyroid',
        'breast': 'breast',
        'skin': 'skin',
        'salivary_gland': 'salivary_gland',

        # --- Capitalization / Spelling Differences ---
        'Colon': 'colon',
        'Placenta': 'placenta',
        'oesophagus': 'esophagus',
        'Testes': 'testis',

        # --- Anatomical / Closest Biological Matches ---
        'urinary_tract': 'bladder',                       # Bladder is the primary tissue of the urinary tract
        'endometrium': 'uterus',                          # Endometrium is the lining of the uterus
        'sinonasal': 'head_and_neck',                     # Sinonasal tract is categorized under head and neck
        'central_nervous_system': 'brain',                # Brain is the primary CNS tissue in the origin set
        'autonomic_ganglia': 'nerve',                     # Ganglia are nerve cell clusters
        'uvea': 'eye',                                    # Uvea is the pigmented layer of the eye
        'Embryonal': 'stem_cell',                         # Embryonal cells map closest to stem cells
        'lymph_node': 'lymphoid',                         # Lymph nodes are made of lymphoid tissue
        'large_intestine': 'colon',                       # Colon is the main segment of the large intestine
        'pleural_effusion': 'pleura',                     # Fluid buildup derived from the pleura
        'pericardial_effusion': 'heart',                  # Fluid buildup around the heart
        'upper_aerodigestive_tract': 'head_and_neck',     # Standard classification for this tract
        'haematopoietic_and_lymphoid_tissue': 'lymphoid', # Mapped to lymphoid (could also be blood/bone_marrow)
        'bone': 'bone_marrow',                            # Closest bone-related tissue available in origin

        # --- No Direct Match / Unspecific ---
        'matched_normal_tissue': None,                    # Context-dependent, no specific tissue
        'ascites': None,                                  # Fluid in the peritoneal cavity, no specific tissue origin
        'abdomen': None,                                  # Too broad (contains stomach, liver, colon, etc.)
        'Unknown': None                                  # No tissue specified
    }

    #claude Tissue Mapping via claude Sonnet 5
    tissue_mapping_claude = {
        # --- Exact / near-exact matches ---
        'stomach': 'stomach',
        'biliary_tract': 'biliary_tract',
        'soft_tissue': 'soft_tissue',
        'thyroid': 'thyroid',
        'breast': 'breast',
        'ovary': 'ovary',
        'lung': 'lung',
        'liver': 'liver',
        'salivary_gland': 'salivary_gland',
        'fibroblast': 'fibroblast',
        'small_intestine': 'small_intestine',
        'spleen': 'spleen',
        'kidney': 'kidney',
        'eye': 'eye',
        'prostate': 'prostate',
        'cervix': 'cervix',
        'bone_marrow': 'bone_marrow',
        'skin': 'skin',
        'pancreas': 'pancreas',
        'pleura': 'pleura',

        # --- Capitalization / spelling fixes ---
        'Testes': 'testis',
        'oesophagus': 'esophagus',
        'Colon': 'colon',
        'Placenta': 'placenta',

        # --- Anatomical / closest biological matches ---
        'pericardial_effusion': 'heart',
        'uvea': 'eye',
        'sinonasal': 'head_and_neck',
        'upper_aerodigestive_tract': 'head_and_neck',
        'large_intestine': 'colon',
        'lymph_node': 'lymphoid',
        'endometrium': 'uterus',
        'urinary_tract': 'bladder',
        'central_nervous_system': 'brain',   # see note below re: spinal_cord

        # --- Judgment calls — flagged, see notes ---
        'autonomic_ganglia': 'nerve',        # see note
        'haematopoietic_and_lymphoid_tissue': 'blood',  # see note
        'bone': 'other',                     # see note
        'Embryonal': 'other',                # see note

        # --- No specific tissue / unspecific ---
        'ascites': 'other',
        'abdomen': 'other',
        'matched_normal_tissue': 'other',
        'Unknown': 'other',
        'pleural_effusion': 'pleura',
    }

    oncotree_to_uberon = {
        "Adrenal Gland": "adrenal_gland",
        "Ampulla of Vater": "biliary_tract",
        "Biliary Tract": "biliary_tract",
        "Bladder/Urinary Tract": "bladder",
        "Bone": None,  # Note: 'bone' is not in the provided Uberon list
        "Bowel": "small_intestine",
        "Breast": "breast",
        "Cervix": "cervix",
        "CNS/Brain": "brain",
        "Esophagus/Stomach":  "stomach",
        "Eye": "eye",
        "Fibroblast": "fibroblast",
        "Head and Neck": "head_and_neck",
        "Kidney": "kidney",
        "Liver": "liver",
        "Lung": "lung",
        "Lymphoid": "lymphoid",
        "Myeloid": "bone_marrow",
        "None": None,
        "Normal": None,
        "Other": None,
        "Ovary/Fallopian Tube": "ovary",
        "Pancreas": "pancreas",
        "Peripheral Nervous System": "nerve",
        "Pleura": "pleura",
        "Prostate": "prostate",
        "Skin": "skin",
        "Soft Tissue": "soft_tissue",
        "Testis": "testis",
        "Thyroid": "thyroid",
        "Uterus": "uterus",
        "Vulva/Vagina": "vagina",
    }

    oncotree_Primdisease_to_uberon = {
        "Acute Leukemias of Ambiguous Lineage": "bone_marrow",
        "Acute Myeloid Leukemia": "bone_marrow",
        "Adenosquamous Carcinoma of the Pancreas": "pancreas",
        "Adrenocortical Carcinoma": "adrenal_gland",
        "Ampullary Carcinoma": "biliary_tract",
        "Anaplastic Thyroid Cancer": "thyroid",
        "B-Lymphoblastic Leukemia/Lymphoma": "bone_marrow", #
        "Bladder Squamous Cell Carcinoma": "bladder",
        "Bladder Urothelial Carcinoma": "bladder",
        "Breast Ductal Carcinoma In Situ": "breast",
        "Breast Neoplasm, NOS": "breast",
        "Cervical Adenocarcinoma": "cervix",
        "Cervical Squamous Cell Carcinoma": "cervix",
        "Chondrosarcoma": "soft_tissue",                #
        "Chordoma": "soft_tissue",                       #
        "Colorectal Adenocarcinoma": "colon",
        "Cutaneous Squamous Cell Carcinoma": "skin",      #
        "Diffuse Glioma": "brain",
        "Embryonal Tumor": "brain",                      #
        "Endometrial Carcinoma": "uterus",
        "Epithelioid Sarcoma": "soft_tissue",            #
        "Esophageal Squamous Cell Carcinoma": "esophagus",
        "Esophagogastric Adenocarcinoma": "stomach",
        "Ewing Sarcoma": "bone_marrow",
        "Extra Gonadal Germ Cell Tumor": "soft_tissue",   #
        "Fibrosarcoma": "soft_tissue",                   #
        "Gastrointestinal Stromal Tumor": "stomach",      #
        "Gestational Trophoblastic Disease": "placenta",
        "Glassy Cell Carcinoma of the Cervix": "cervix",
        "Head and Neck Carcinoma, Other": "head_and_neck",
        "Head and Neck Squamous Cell Carcinoma": "head_and_neck",
        "Hepatoblastoma": "liver",
        "Hepatocellular Carcinoma": "liver",
        "Hepatocellular Carcinoma plus Intrahepatic Cholangiocarcinoma": "liver",
        "Hodgkin Lymphoma": "lymphoid",
        "Intracholecystic Papillary Neoplasm": "biliary_tract",                #
        "Intraductal Papillary Neoplasm of the Bile Duct": "biliary_tract",
        "Invasive Breast Carcinoma": "breast",
        "Leiomyosarcoma": "muscle",
        "Liposarcoma": "adipose",
        "Lung Neuroendocrine Tumor": "lung",
        "Medullary Thyroid Cancer": "thyroid",
        "Melanoma": "skin",
        "Meningothelial Tumor": "brain",
        "Merkel Cell Carcinoma": "skin",
        "Mixed Cervical Carcinoma": "cervix",
        "Mucosal Melanoma of the Vulva/Vagina": "vagina",
        "Myelodysplastic Syndromes": "bone_marrow",
        "Myeloproliferative Neoplasms": "bone_marrow",
        "Nerve Sheath Tumor": "nerve",
        "Neuroblastoma": "adrenal_gland",
        "Non-Cancerous": None,
        "Non-Hodgkin Lymphoma": "lymphoid",
        "Non-Seminomatous Germ Cell Tumor": "testis",
        "Non-Small Cell Lung Cancer": "lung",
        "Ocular Melanoma": "eye",
        "Osteosarcoma": "bone_marrow",
        "Ovarian Epithelial Tumor": "ovary",
        "Ovarian Germ Cell Tumor": "ovary",
        "Ovarian Cancer, Other": "ovary",
        "Pancreatic Adenocarcinoma": "pancreas",
        "Pancreatic Neuroendocrine Tumor": "pancreas",
        "Pleural Mesothelioma": "pleura",
        "Poorly Differentiated Thyroid Cancer": "thyroid",
        "Prostate Adenocarcinoma": "prostate",
        "Prostate Small Cell Carcinoma": "prostate",
        "Renal Cell Carcinoma": "kidney",
        "Retinoblastoma": "eye",
        "Rhabdoid Cancer": None,
        "Rhabdomyosarcoma": "muscle",
        "Salivary Carcinoma": "salivary_gland",
        "Sarcoma, NOS": "soft_tissue",
        "Sex Cord Stromal Tumor": "ovary",
        "Small Bowel Cancer": "small_intestine",
        "Small Cell Carcinoma of the Cervix": "cervix",
        "Squamous Cell Carcinoma of the Vulva/Vagina": "vagina",
        "Synovial Sarcoma": "soft_tissue",
        "T-Lymphoblastic Leukemia/Lymphoma": "bone_marrow",
        "Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma": "soft_tissue",
        "Urethral Cancer": None,
        "Uterine Sarcoma/Mesenchymal": "uterus",
        "Well-Differentiated Thyroid Cancer": "thyroid",
        None: None,
    }

    oncotree_Primdisease_to_uberon_revised = {
        "Acute Leukemias of Ambiguous Lineage": "bone_marrow",
        "Acute Myeloid Leukemia": "bone_marrow",
        "Adenosquamous Carcinoma of the Pancreas": "pancreas",
        "Adrenocortical Carcinoma": "adrenal_gland",
        "Ampullary Carcinoma": "biliary_tract",
        "Anaplastic Thyroid Cancer": "thyroid",
        "B-Lymphoblastic Leukemia/Lymphoma": "bone_marrow",
        "Bladder Squamous Cell Carcinoma": "bladder",
        "Bladder Urothelial Carcinoma": "bladder",
        "Breast Ductal Carcinoma In Situ": "breast",
        "Breast Neoplasm, NOS": "breast",
        "Cervical Adenocarcinoma": "cervix",
        "Cervical Squamous Cell Carcinoma": "cervix",
        "Chondrosarcoma": "soft_tissue",
        "Chordoma": "soft_tissue",
        "Colorectal Adenocarcinoma": "colon",
        "Cutaneous Squamous Cell Carcinoma": "skin",
        "Diffuse Glioma": "brain",
        "Embryonal Tumor": "brain",
        "Endometrial Carcinoma": "uterus",
        "Epithelioid Sarcoma": "soft_tissue",
        "Esophageal Squamous Cell Carcinoma": "esophagus",
        "Esophagogastric Adenocarcinoma": None,
        "Ewing Sarcoma": "soft_tissue",
        "Extra Gonadal Germ Cell Tumor": None,
        "Fibrosarcoma": "soft_tissue",
        "Gastrointestinal Stromal Tumor": None,
        "Gestational Trophoblastic Disease": "placenta",
        "Glassy Cell Carcinoma of the Cervix": "cervix",
        "Head and Neck Carcinoma, Other": "head_and_neck",
        "Head and Neck Squamous Cell Carcinoma": "head_and_neck",
        "Hepatoblastoma": "liver",
        "Hepatocellular Carcinoma": "liver",
        "Hepatocellular Carcinoma plus Intrahepatic Cholangiocarcinoma": "liver",
        "Hodgkin Lymphoma": "lymphoid",
        "Intracholecystic Papillary Neoplasm": "biliary_tract",
        "Intraductal Papillary Neoplasm of the Bile Duct": "biliary_tract",
        "Invasive Breast Carcinoma": "breast",      # ______________________________________________________________________________________________________
        "Leiomyosarcoma": "muscle",
        "Liposarcoma": "adipose",
        "Lung Neuroendocrine Tumor": "lung",
        "Medullary Thyroid Cancer": "thyroid",
        "Melanoma": "skin",
        "Meningothelial Tumor": "brain",
        "Merkel Cell Carcinoma": "skin",
        "Mixed Cervical Carcinoma": "cervix",
        "Mucosal Melanoma of the Vulva/Vagina": None,
        "Myelodysplastic Syndromes": "bone_marrow",
        "Myeloproliferative Neoplasms": "bone_marrow",
        "Nerve Sheath Tumor": "nerve",
        "Neuroblastoma": "adrenal_gland",
        "Non-Cancerous": None,
        "Non-Hodgkin Lymphoma": "lymphoid",
        "Non-Seminomatous Germ Cell Tumor": "testis",
        "Non-Small Cell Lung Cancer": "lung",
        "Ocular Melanoma": "eye",
        "Osteosarcoma": None,
        "Ovarian Epithelial Tumor": "ovary",
        "Ovarian Germ Cell Tumor": "ovary",
        "Ovarian Cancer, Other": "ovary",
        "Pancreatic Adenocarcinoma": "pancreas",
        "Pancreatic Neuroendocrine Tumor": "pancreas",
        "Pleural Mesothelioma": "pleura",
        "Poorly Differentiated Thyroid Cancer": "thyroid",
        "Prostate Adenocarcinoma": "prostate",
        "Prostate Small Cell Carcinoma": "prostate",
        "Renal Cell Carcinoma": "kidney",
        "Retinoblastoma": "eye",
        "Rhabdoid Cancer": None,
        "Rhabdomyosarcoma": "muscle",
        "Salivary Carcinoma": "salivary_gland",
        "Sarcoma, NOS": "soft_tissue",
        "Sex Cord Stromal Tumor": None,
        "Small Bowel Cancer": "small_intestine",
        "Small Cell Carcinoma of the Cervix": "cervix",
        "Squamous Cell Carcinoma of the Vulva/Vagina": None,
        "Synovial Sarcoma": "soft_tissue",
        "T-Lymphoblastic Leukemia/Lymphoma": "bone_marrow",
        "Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma": "soft_tissue",
        "Urethral Cancer": None,
        "Uterine Sarcoma/Mesenchymal": "uterus",
        "Well-Differentiated Thyroid Cancer": "thyroid",
        None: None,
        }
    c = apply_tissue_mapping(pl.read_csv(comparpth), oncotree_Primdisease_to_uberon_revised, target_col, original_col).filter(pl.col(original_col).is_not_null())
    #c = c.filter(pl.col(original_col).is_not_null())
    #c = c.select([pl.col("ModelID"),pl.col(original_col)])
    c = c.filter(pl.col(samplecol).is_in(samples_in_modality(c,pl.read_csv(gexpth),samplecol)))
    print(c.columns)
    print(c.head())
    input("Press Enter to continue...")
    c.write_csv(savleloc)
