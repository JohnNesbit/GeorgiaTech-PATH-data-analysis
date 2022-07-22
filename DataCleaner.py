import pandas as pd
import numpy as np
import os

# import csvs into pandas dataframes, combine them
path = "ICPSR_36498-V16 (1)/ICPSR_36498" # put path to list of "DS" directories of "deliniated" version
wantedCols = ["R04_EVER_NEVER_CIGS"] # wanted columns of information

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
		nonSmokerDf = df
	elif(len(nonSmokerDf) > 10): # add onto dir once initialized, for some reason I had to add the elif, dont ask
		# get columns we want
		for wcol in wantedCols:
			if (wcol in df.columns):
				nonSmokerDf.merge(df["PERSONID", wcol], on="PERSONID", how='left') # merge dataframes, they have new data on same people


	if(_ > 3): # if you want to test, only uses 3 dataframes so not to take 2 years to run and all 8 gigs of RAM
		break
	_ += 1


# delete those who have smoked cigarettes before
import gc
nonSmokerDf = nonSmokerDf["R04_EVER_NEVER_CIGS"] == 2 # 2 is never user - DS4001/36498-4001-Questionnaire.pdf pg 25
gc.collect()

cols = nonSmokerDf.columns
print("cols: ", len(cols))
print("rows: ", len(nonSmokerDf))
for i in cols: # print the cols we kept, ensure everything is there
	print(i)

# change NULLs to 0
# only on questions where this makes sense
yes_or_no_questions = []
nonSmokerDf[yes_or_no_questions][nonSmokerDf[yes_or_no_questions] != 1 and nonSmokerDf[yes_or_no_questions] != 2] = 0

# create csv and save filtered data"""
nonSmokerDf.to_csv("Cleaned_data.csv")
