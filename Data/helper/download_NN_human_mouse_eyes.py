import requests
import zipfile
import io
import sys
from clint.textui import progress

file = 'NN_human_mouse_eyes.zip'
folder = 'NN_human_mouse_eyes'
url_string = 'https://zenodo.org/record/4488164/files/NN_human_mouse_eyes.zip?download=1'



r = requests.get(url_string, stream=True)
with open(file, 'wb') as f:
    total_length = int(r.headers.get('content-length'))
    for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
        if chunk:
            f.write(chunk)
            f.flush()

# with zipfile.ZipFile(file, 'r') as zip_ref:
#     zip_ref.extractall(folder)