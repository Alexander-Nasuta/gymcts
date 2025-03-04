import nbformat
import os

def add_header_to_notebook(notebook_path, header_text):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    header_cell = nbformat.v4.new_markdown_cell(f'# {header_text}')
    nb.cells.insert(0, header_cell)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def add_header_to_all_notebooks(folder_path, header_text):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                add_header_to_notebook(notebook_path, header_text)

# Usage
folder_path = './'
header_text = 'Your Header Title'
add_header_to_all_notebooks(folder_path, header_text)

if __name__ == '__main__':
    pass