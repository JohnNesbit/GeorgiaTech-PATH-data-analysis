import pandas as pd
import numpy as np

# import csvs into pandas dataframes, combine them

# change NULLs to 0

# split data into people who have smoked cigarettes and those who have not

# delete those who have smoked cigarettes before
import gc
# del SmokerDf
gc.collect()

# change "definitely yes" and "probably yes" answers into just "yes". Do the same for no

# filter wanted questions
cols = []
nonSmokers = nonSmokers[cols]

# encode questions into numerical buckets -  null:0 yes:1 no:2
nonSmokers[col][nonSmokers[col] == "yes"] = 1 # I believe this is the correct syntax for pandas dataframe stuffs
nonSmokers[col][nonSmokers[col] == "yes"] = 2

# create csv and save filtered data
