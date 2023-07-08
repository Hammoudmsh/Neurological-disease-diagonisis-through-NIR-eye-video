
import requests
from clint.textui import progress
import zipfile
import io
import sys

file = 's-natural.zip'
folder = 's-natural'
url_string = 'https://cs.rit.edu/~cgaplab/RIT-Eyes/official_release/s-natural.zip'





r = requests.get(url_string, stream=True)
with open(file, 'wb') as f:
    total_length = int(r.headers.get('content-length'))
    for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
        if chunk:
            f.write(chunk)
            f.flush()

# with zipfile.ZipFile(file, 'r') as zip_ref:
#     zip_ref.extractall(folder)