import requests
import zipfile
import io
import sys
import shutil


with zipfile.ZipFile('s-openeds.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
shutil.copy2('color_maps/classes_s-openeds.json', 's-openeds/classes.json') # target filename is /dst/dir/file.ext


with zipfile.ZipFile('s-nvgaze.zip', 'r') as zip_ref:
    zip_ref.extractall('s-nvgaze')
shutil.copy2('color_maps/classes_s-nvgaze.json', 's-nvgaze/classes.json') # target filename is /dst/dir/file.ext

with zipfile.ZipFile('s-natural.zip', 'r') as zip_ref:
    zip_ref.extractall('s-natural')
shutil.copy2('color_maps/classes_s-natural.json', 's-natural/classes.json') # target filename is /dst/dir/file.ext


# with zipfile.ZipFile('DataSet.zip', 'r') as zip_ref:
#     zip_ref.extractall('DataSet')
    
with zipfile.ZipFile('ClinicAnnotated.zip', 'r') as zip_ref:
    zip_ref.extractall('ClinicAnnotated')






