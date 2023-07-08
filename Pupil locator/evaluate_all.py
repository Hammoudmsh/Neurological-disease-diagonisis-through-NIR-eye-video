import os
import pathlib
import datetime

INTERPRETER = 'python'

def myTasks(out, model_path, CLASSES_NUM, WANTED):
	os.system(f"{INTERPRETER} eval.py --DS_NAME ClinicAnnotated_DA --OUTPUT_PATH '{out}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --MODEL_PATH '{model_path}' --SEGMENT 1 --imgExt 'png' --maskExt 'png'")
	print("________________________________________________________________________________________________")
	os.system(f"{INTERPRETER} eval.py --DS_NAME NN_human_mouse_eyes --OUTPUT_PATH '{out}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --MODEL_PATH '{model_path}' --SEGMENT 1 --imgExt 'jpg' --maskExt 'png'")
	print("________________________________________________________________________________________________")
	#os.system(f"{INTERPRETER} eval.py --DS_NAME MOBIUS --OUTPUT_PATH '{out}' --MODEL_PATH '{model_path}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --SEGMENT 1 --imgExt png --maskExt png")
	#print("________________________________________________________________________________________________")
	os.system(f"{INTERPRETER} eval.py --DS_NAME s-openeds --OUTPUT_PATH '{out}' --MODEL_PATH '{model_path}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --SEGMENT 1 --imgExt 'tif' --maskExt 'tif'")
	print("________________________________________________________________________________________________")
	os.system(f"{INTERPRETER} eval.py --DS_NAME s-nvgaze --OUTPUT_PATH '{out}' --MODEL_PATH '{model_path}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --SEGMENT 1 --imgExt 'tif' --maskExt 'tif'")
	print("________________________________________________________________________________________________")
	os.system(f"{INTERPRETER} eval.py --DS_NAME s-natural --OUTPUT_PATH '{out}' --MODEL_PATH '{model_path}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --SEGMENT 1 --imgExt 'tif' --maskExt 'tif'")
	#os.system(f"{INTERPRETER} eval.py --DS_NAME IITD_database --OUTPUT_PATH '{out}' --MODEL_PATH '{model_path}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --SEGMENT 0 --imgExt 'bmp'")
	#print("________________________________________________________________________________________________")
	#os.system(f"{INTERPRETER} eval.py --DS_NAME MMU-Iris-Database --OUTPUT_PATH '{out}' --MODEL_PATH '{model_path}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --SEGMENT 0 --imgExt 'bmp'")
	#print("________________________________________________________________________________________________")
	#os.system(f"{INTERPRETER} eval.py --DS_NAME LPW --OUTPUT_PATH '{out}' --MODEL_PATH '{model_path}' --CLASSES_NUM {CLASSES_NUM} --WANTED '{WANTED}' --SEGMENT 0 --imgExt 'png' --maskExt 'png'")
	#print("________________________________________________________________________________________________")
	os.system(f"{INTERPRETER} plotSamples.py")
	#print("________________________________________________________________________________________________")
 
if __name__ == "__main__":
#global out
#current_model = datetime.datetime.now().strftime("%d_%m_%Y(%H_%M_%S)")
#out = f'../Results/evaluation/{current_model}/'
#pathlib.Path(out).mkdir(parents=True, exist_ok=True)
#myTasks(out, "../Results/model1/model_mine_10_02_2023:10_35_47_openeds_mnet_p_100_last_m/model1_results_segs_best.hdf5", 2, 'p')
  
  
  models = {
  #"m1":"model_mine_09_02_2023:16_29_02_clinin_mnet_p_125_last_m",
  #"m2":"model_mine_09_02_2023:23_21_38_NN_mnet_p_100_last_m",
  #"m3":"model_mine_10_02_2023:18_01_16_clinic_NN_both_sk_last_50_100",
  "m4":"model_mine_10_02_2023:10_35_47_openeds_mnet_p_100_last_m",
  #"m5":"model_mine_11_02_2023:10_31_07_clinic_openeds_both_sk_last_50_100",
  #"m6":"model_mine_10_02_2023:21_32_17_NN_Openeds_both_sk_last_10_100_m",
  #"m7":"model_mine_11_02_2023:01_13_41_Clinic_NN_Openeds_both_sk_last_25_10_100_m"
  }
  
  current_model = "res1"#datetime.datetime.now().strftime("%d_%m_%Y(%H_%M_%S)")
  for k, m in models.items():
    out = f'../Results/evaluation/{current_model}/{k}/'
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    myTasks(out, f"../Results/model1/{m}/model1_results_segs_best.hdf5", 2, 'p')