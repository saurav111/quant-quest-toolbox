import pandas as pd
import numpy as np

import xgboost as xgb #pip install xgboost
import os

from sklearn.model_selection import train_test_split

csvs = os.listdir("abcd/historicalData")
train = [pd.read_csv(os.path.join("abcd/historicalData",x),index_col=0) for x in csvs]

train_numpy = [np.array(x) for x in train] 
label_vecs = [np.sign(sth[0:sth.shape[0]-1,0] - sth[1:sth.shape[0],3]) for sth in train_numpy] #correct labels
final_train_sets = [x[1:x.shape[0],:] for x in train_numpy] #can't look into the future


X = np.vstack(final_train_sets[0:len(final_train_sets)-1])
y = np.hstack(label_vecs[0:len(label_vecs)-1])
y[y==-1] = 0

clf = xgb.XGBClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = xgb.XGBClassifier()


print("Training the model...")
clf.fit(X_train, y_train)

print("Test score is")
print(clf.score(X_test,y_test))
#Close to random right now

# Ideas for improvement 
# Cross validation and parameter tuning
# Add the date feature
# Keep separate models for separate sectors
