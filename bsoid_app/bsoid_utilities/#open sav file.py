# open sav file
import joblib
with open('C:\\Users\\chang\\DeepLabCut\\main\\JUPYTER\\DLC_Data\\output\\test_clusters.sav', 'rb') as fr: 
   data = joblib.load(fr)
   print(type(data[2]))
   print(data[2].shape)

# save data as csv file
# import pandas as pd
# df = pd.DataFrame(data[0])
# df.to_csv('C:\\Users\\chang\\DeepLabCut\\main\\JUPYTER\\DLC_Data\\output\\test_feats.csv', index=False)


# map cluster labels to videos