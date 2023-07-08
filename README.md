# Application of deep learning methods for the investigation of the neurological disease diagnosed through NIR eye video

## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [requirements](#requirements)
* [Launch](#launch)
* [Results](#results)

## Introduction
Deep learning is extensively used in neurological applications as decision support for doctors. Parkin disease(PD) and Progressive Supranuclear Palsy(PSP) are crucial neurodegenerative diseases that have similar symptoms. The goal of this work is to classify PD, PSP, and Healthy Control(HC) with high accuracy through eye movements. The paper is <a href="" target="_blank">Paper(will be added)</a>.


## Technologies
* Computer vision
* Deep learning
* Semantic segmentation
* time-series analysis

## Requirements
1- create a virtual environment with conda and activate it
```
conda create --name sk_pro python=3.6
conda activate sk_pro
pip install -r requirements_paper.txt
```
## Launch
### download data

* Manual downloading:
<a href="https://drive.google.com/drive/folders/1qFETu_crBA0_gBViBfHNQkZmlK1e_mou?usp=sharing" target="_blank">Patients</a>, <a href="https://cs.rit.edu/~cgaplab/RIT-Eyes/" target="_blank">OpenEDS</a>, <a href="https://cs.rit.edu/~cgaplab/RIT-Eyes/" target="_blank">Navgaze</a>, <a href="https://zenodo.org/record/4488164#.Y7U7YdVByUl" target="_blank">NN human mouse eyes</a>

* Partial automatic run scripts, to download opened, navgaze, natural, and NN human mouse datasets.
```
cd Data/
python Data/download_data.py
```
- download the <a href="https://drive.google.com/drive/folders/1qFETu_crBA0_gBViBfHNQkZmlK1e_mou?usp=sharing" target="_blank">Patients</a> dataset(manually) and add zip file into Data

### Extract data
```
python Data/extract_data.py
```


### Pupil segmentation model
1- augment the data
```
cd ../Pupil locator/
python DS_DA.py
```
2 - Commands
* train
```
python train.py --DS_NAME 'NN_human_mouse_eyes' --epochs 1 --batch_size 128 --lr 1e-05 --es 1 --wanted 'p' --file2read 100 --backbon 'mine' --weights '[1,0,0,0]' --output output_id
```
* Inference on image
```
python eval.py --IMG_PATH ../toBeTested/14.png  --OUTPUT_PATH test_results/total_red_border1.png --MODEL_PATH model1_results_segs_best.hdf5 --CLANE 1
python eval.py --IMG_PATH  ../toBeTested/d21.bmp ../toBeTested/d22.bmp ../toBeTested/d31.png ../toBeTested/d32.png ../toBeTested/d41.tif ../toBeTested/d42.tif --OUTPUT_PATH test_results/total_red_border2.png --MODEL_PATH model1_results_segs_best.hdf5 --CLANE 1
python eval.py --IMG_PATH  ../toBeTested/d21.bmp ../toBeTested/d22.bmp ../toBeTested/d31.png ../toBeTested/d32.png ../toBeTested/d41.tif ../toBeTested/d42.tif --OUTPUT_PATH test_results/total_red_border2.png --MODEL_PATH model1_results_segs_best.hdf5 --CLANE 1
```
* Inference on video
```
python eval.py --VIDEO_PATH ../toBeTested/test_video1.avi --OUTPUT_PATH to/ --MODEL_PATH model1_results_segs_best.hdf5
```
* Evaluate on datasets(in case you need to evaluate on others datasets)
```
python evaluate_all.py
```
Note: in case you need to extract frames from video
```
python DS_clinicDS2Frames.py
```
### Diseases classification model
1- Extract the pupil's properties from the video dataset. It takes a long time, so no need to run, we already provided the extracted features in CSV files. in case of further enhancements, produce a new model for pupil extraction and run this command.
```
python feature_extractor.py
```
2- Transform time-series features(signals) into images.
```
python featuresToImage.py
```
3- training
```
python train1.py --epochs 1000 --lr 0 --file2read 0 --es 1 --batch_size 32 --WANTED_TESTS 'P' --WANTED_FEATURES 'YAJ' --TS 500 --SIZE 90 --WANTED_ALGS 'GADF' --USED_MODEL_ARCH CNN --output output_id
```
# Results
## **Puipil segmentation model: Here are the results of inference on some images from different datasets.**
![image](https://github.com/Hammoudmsh/Neurological-disease-diagonisis-through-eye-movements/assets/57059181/da381039-1069-4e43-b537-d98d35c5a0b9)
***<p style="text-align: center;">   </p>***
<br />
![image](https://github.com/Hammoudmsh/Neurological-disease-diagonisis-through-eye-movements/assets/57059181/fe8558bf-8940-4bbc-b41c-db9744d0edda)
<br />
![image](https://github.com/Hammoudmsh/Neurological-disease-diagonisis-through-eye-movements/assets/57059181/17f25b51-6e9f-4f75-a658-5ef2db4e2594)
<br />
![image](https://github.com/Hammoudmsh/Neurological-disease-diagonisis-through-eye-movements/assets/57059181/6772c759-a321-4e2d-820f-8d78b8c7d2e3)
<br />
## **Confusion matrix of disease classification from left, right, and both eyes**
![image](https://github.com/Hammoudmsh/Neurological-disease-diagonisis-through-NIR-eye-video/assets/57059181/e972af9c-9eea-4871-b1d3-9a08788c64fa)


