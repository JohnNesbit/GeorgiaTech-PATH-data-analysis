import pandas as pd
import numpy as np
import os

# import csvs into pandas dataframes, combine them
path = "ICPSR_36498-V16 (1)/ICPSR_36498" # put path to list of "DS" directories of "deliniated" version

# find tsv file in the directory given
def findData(path):
	for file in os.listdir(path):
		if("-Data.tsv" in file):
			return path + file

waves = {}
for i in range(1, 6): # loop through waves of questionarres
	for j in range(1, 3): # j determines adult or youth questionarre file, Youth is 2
		# this line loads the given wave and age group into the dict "waves" with a correspoding hash
		waves[str(i) + "_" + str(j)] = pd.read_csv(findData("E:/PycharmProjects/Gt-data-science-project/ICPSR_36498-V16 (1)/ICPSR_36498/DS"+ str(i) +"00"+ str(j) +"/"), sep='\t', low_memory=False)


# get people who were new smokers in the wave and get their previous wave data for prediction
getSmokers = lambda wave, var, previousWave: previousWave[np.isin(previousWave["PERSONID"], wave[wave[var] == 1]["PERSONID"])]

# get youth who have smoked since the first wave
# include age-up adults by finding the new smokers and then only findiing matches within the last wave's youth files
adict = {1:"A", 2:"Y"}
for i in range(2, 6):
	for j in range(1, 3):
		if (i == 2 and j == 1):
			smokers = getSmokers(waves["2_1"], "R02R_A_NEW_CIGS", waves["1_2"])
			continue
		smokers = smokers.append(getSmokers(waves[str(i) + "_" + str(j)], "R0"+ str(i) +"R_"+ adict[j] +"_NEW_CIGS", waves[str(i-1) + "_2"]), sort=True)

print(len(smokers)) # 1824 youths started smoking during this experiment

for col in smokers.columns: # have to go through cols manually instead of np.where because NaNs must be excepted
	if col == "PERSONID" or col == "CASEID":
		continue

	try:
		smokers[col] = pd.to_numeric(smokers[col], downcast="integer")
	except:
		smokers = smokers.drop(columns=[col])
		print(col)
		continue

	smokers[col] = np.where(np.nan_to_num(smokers[col]) < 0, np.nan, smokers[col])

nanlist = [] # linked list for nans and columns
for col in smokers.columns:
	nanlist.append(smokers[col].isnull().sum()) # count nans in each column

print(smokers.columns[np.argmax(np.array(nanlist))], ": ", max(nanlist)) # see how many nans are within highest var
# was 1428/1824. clearly not a relevant or acceptable variable

print(len(smokers.columns))
colist = []
for l in nanlist:
	if l >= 1:
		colist.append(smokers.columns[l])

smokers = smokers.drop(columns=colist)
print(len(smokers.columns))
