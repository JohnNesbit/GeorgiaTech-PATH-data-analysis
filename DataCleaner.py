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
		cols = list(waves[str(i) + "_" + str(j)].columns)
		for l in range(len(cols)):
			if "R" == cols[l][0]:
				cols[l] = cols[l][3:]
		waves[str(i) + "_" + str(j)].columns = cols


# get people who were new smokers in the wave and get their previous wave data for prediction
getSmokers = lambda wave, var, previousWave: previousWave[np.isin(previousWave["PERSONID"], wave[wave[var] == 1]["PERSONID"])]

# get youth who have smoked since the first wave
# include age-up adults by finding the new smokers and then only findiing matches within the last wave's youth files
adict = {1:"A", 2:"Y"}
for i in range(2, 6):
	for j in range(1, 3):
		if i == 2 and j == 1:
			smokers = getSmokers(waves["2_1"], "R_A_NEW_CIGS", waves["1_2"]).copy()
			print(len(smokers))
			print(len([i for l in range(len(smokers))]))
			smokers["WAVE"] = [i for l in range(len(smokers))]# create wave variable for the time-period they are in
			# this wave variable is mainly for balancing the dataset
			continue
		sm = getSmokers(waves[str(i) + "_" + str(j)], "R_"+ adict[j] +"_NEW_CIGS", waves[str(i-1) + "_2"]).copy()
		sm["WAVE"] = [i for l in range(len(sm))]
		smokers = smokers.append(sm, sort=True) # create wave variable

print(len(smokers)) # 1824 youths started smoking during this experiment
smokers["Target"] = list(np.ones(len(smokers))) # mark smokers as smokers with extra variable

for i in range(2, 6): # loop through waves
	add = waves[str(i) + "_2"][np.logical_not(np.isin(waves[str(i) + "_2"]["PERSONID"], smokers["PERSONID"]))].copy() # ensure no duplicates
	add = add[add["R_Y_EVR_CIGS"] == 2]
	add = add[:smokers["WAVE"].tolist().count(i)]  # add matching data from each wave
	add["Target"] = list(np.zeros(len(add))) # set smokers to false for the new data
	add["WAVE"] = [i for l in range(len(add))]

	smokers = smokers.append(add, sort=True) # append to smokers
	print(len(smokers))
	# after this data has 50% of people who become smokers, 50% who do not

for col in smokers.columns: # have to go through cols manually instead of np.where because strings must be excepted
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
	if col == "PERSONID" or col == "CASEID":
		continue

	nanlist.append(np.isnan(smokers[col]).sum()) # count nans in each column

print(smokers.columns[np.argmax(np.array(nanlist))], ": ", max(nanlist)) # see how many nans are within highest var
print(nanlist)
# was 1428/1824. clearly not a relevant or acceptable variable

# drop all cols with nans
print(len(smokers.columns))
colist = []
for _, l in enumerate(nanlist):
	if l >= 2000:
		colist.append(smokers.columns[2+_])

smokers = smokers.drop(columns=colist)
print(len(smokers.columns))

for i, ii in zip(smokers["WAVE"], smokers["Target"]):
	print(i, ": ", ii)

# roughly 3600 People across 5 waves, 4700 cleaned variables
smokers.to_csv("DirtyData2000.csv") # save data
