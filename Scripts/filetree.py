from pathlib import Path

def generate_tree(path: Path, prefix: str = "") -> str:
    # Filter out hidden files/folders if desired
    contents = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
    pointers = [("├── " if i < len(contents) - 1 else "└── ") for i in range(len(contents))]
    
    tree_str = ""
    for pointer, path_obj in zip(pointers, contents):
        is_dir = path_obj.is_dir()
        tree_str += prefix + pointer + path_obj.name + ("/" if is_dir else "") + "\n"
        if is_dir:
            extension = "│   " if pointer == "├── " else "    "
            tree_str += generate_tree(path_obj, prefix + extension)
    return tree_str

def save_tree_to_txt(target_dir=".", output_filename="directory_tree.txt"):
    root_path = Path(target_dir).resolve()
    content = root_path.name + "/\n" + generate_tree(root_path)
    
    output_path = Path(output_filename)
    output_path.write_text(content, encoding="utf-8")
    print(f"Successfully exported directory tree to: {output_path.absolute()}")

if __name__ == "__main__":
    # Change "." to any target directory path if needed
    save_tree_to_txt(".")