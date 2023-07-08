import requests
from clint.textui import progress
import zipfile
import io
import sys
import os

print(os.getcwd())

os.system('python helper/download_NN_human_mouse_eyes.py')
os.system('python helper/download_openeds.py')
os.system('python helper/download_nvgaze.py')
os.system('python helper/download_natural.py')



