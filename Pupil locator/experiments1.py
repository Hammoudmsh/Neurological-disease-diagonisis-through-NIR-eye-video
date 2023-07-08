import csv
import pandas as pd
import os 
import datetime

df = pd.read_csv('experiments1.csv', index_col=False,sep=" ")
# display(df)


df1 = df[df["yes"]!=0]
#print(df1)
stamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")


cols = df1.columns
for i in range(df1.shape[0]):
    command = ""
    for c in cols:
        if c != "file" and c != "yes" :
            command += f" --{c} {df1.iloc[i][c]}"
    command = df1.iloc[i]["file"] + command
    command = f"nohup python3 {command} > ../Results/logs/{df1.iloc[i]['output']}_{stamp}.out &"
    print(command)
    os.system(command)
    