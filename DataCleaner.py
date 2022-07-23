import pandas as pd
import numpy as np
import os

# import csvs into pandas dataframes, combine them
path = "ICPSR_36498-V16 (1)/ICPSR_36498" # put path to list of "DS" directories of "deliniated" version

nonSmokerDf = []
wantedCols = ["R01_EVER_USER_CIGS"] # wanted columns of information
foundWantedCols = []
with open("ColumnInfo.txt", "r") as f:
	for col in f.read().split(","):
		wantedCols.append(col)

def findData(path):
	for file in os.listdir(path):
		if("-Data.tsv" in file):
			return path + file

_ = 0
for i in os.listdir(path):
	if("DS" not in i):
		continue
	print(findData(path+"/"+i+"/"))
	df = pd.read_csv(findData(path+"/"+i+"/"), sep='\t')

	if(_ == 0): # initialize dir
		for wcol in wantedCols:
			if (wcol in df.columns):
				nonSmokerDf = df[["PERSONID", wcol]] # merge dataframes, they have new data on same people
				foundWantedCols.append(wcol)
	elif(len(nonSmokerDf) > 10): # add onto dir once initialized, for some reason I had to add the elif, dont ask
		# get columns we want
		for wcol in wantedCols:
			if (wcol in df.columns):
				nonSmokerDf.merge(df[["PERSONID", wcol]], on="PERSONID", how='left') # merge dataframes, they have new data on same people
				if(wcol not in foundWantedCols):
					foundWantedCols.append(wcol)


	if(_ > 2): # if you want to test, only uses 3 dataframes so not to take 2 years to run and all 8 gigs of RAM
		break
	_ += 1


# delete those who have smoked cigarettes before
#import gc
#nonSmokerDf = nonSmokerDf[nonSmokerDf["R04_EVER_NEVER_CIGS"] == 2] # 2 is never user - DS4001/36498-4001-Questionnaire.pdf pg 25
#gc.collect()

cols = nonSmokerDf.columns
print("cols: ", len(cols))
print("rows: ", len(nonSmokerDf))
for i in cols: # print the cols we kept, ensure everything is there
	print(i)

for col in foundWantedCols:
	nonSmokerDf.loc[:, (col, nonSmokerDf[col] < 1)] = 0 # get rid of negative answers and turn into 0

# create csv and save filtered data"""
nonSmokerDf.to_csv("Cleaned_data.csv")
