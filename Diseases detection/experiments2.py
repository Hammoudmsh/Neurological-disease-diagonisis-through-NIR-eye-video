import csv
import pandas as pd
import os 
import datetime

df = pd.read_csv('experiments2.csv', index_col=False,sep=",")
df1 = df[df["yes"]!=0]
#df1 = df1.drop("axis = 1)

cols = df1.columns
for i in range(df1.shape[0]):
    command = ""
    stamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    output_name = ""
    for c in cols:
        if c != "file" and c != "yes" :
            if c in ["WANTED_TESTS", "WANTED_FEATURES", "WANTED_ALGS"]:
                command += f" --{c} '{df1.iloc[i][c]}'"
            elif c == "output":
                output_name = f"{df1.iloc[i]['output']}_{stamp}_rowId{i}"
                command += f" --{c} {output_name}"
            else:
                command += f" --{c} {df1.iloc[i][c]}"
                
    command = df1.iloc[i]["file"] + command
    command = f"nohup python3 {command} > ../Results/logs2/{output_name}.out"
    print(command)
    # os.system(command)
