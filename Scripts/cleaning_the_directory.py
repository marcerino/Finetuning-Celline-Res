import os


def all_pth_paths_save_core(headdir):
    pth_paths = []
    for root, dirs, files in os.walk(headdir):
        for file in files:
            if file.endswith(".pth") and not file.endswith("fullfeatures.final_model.pth"):
                pth_paths.append(os.path.join(root, file))
    return pth_paths

def move_and_symlink(pth_paths, headdir, saveloc):
    os.makedirs(saveloc, exist_ok=True)

    for pth_path in pth_paths:
        filename = os.path.basename(pth_path)
        save_path = os.path.join(saveloc, filename)
        os.rename(pth_path, save_path)
        os.symlink(save_path, pth_path)
        

if __name__ == "__main__":
    headdir = os.path.dirname(os.getcwd())
    pth_paths = all_pth_paths_save_core(headdir)
    save_loc = os.path.join(headdir, "BIG_FILES_TO_DISCARD")
    move_and_symlink(pth_paths, headdir, save_loc)