import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# load practice data
practice_data_path = "practice_data"
practice_data_file = os.path.join(practice_data_path, 'NGC_2509.tsv')
df = pd.read_csv(practice_data_file, sep='\t', header=61)

# remove rows between header and data
df = df.iloc[2:]
df = df.reset_index(drop=True)

# lists of columns to filter the dataframe by
essential_columns = ['RA_ICRS', 'DE_ICRS', 'plx', 'pmRA', 'pmDE', 'Gmag', 'BP-RP', 'PMemb']
training_columns = ['RA_ICRS', 'DE_ICRS', 'plx', 'pmRA', 'pmDE', 'Gmag', 'BP-RP']
tgas_training_columns = ['RA_ICRS', 'DE_ICRS', 'plx', 'pmRA', 'pmDE']

# make new dataframe with only the essential columns
df = df[essential_columns]

# convert data to floats
df = df.astype(np.float32)

# new dataframes with only members and non members
probability_threshold = 0.9
members = df[(df['PMemb'] >= probability_threshold)]
non_members = df[(df['PMemb'] < probability_threshold)]

# plot members and non members
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(members['RA_ICRS'], members['DE_ICRS'], label='members')
ax.scatter(non_members['RA_ICRS'], non_members['DE_ICRS'], label='non members')
ax.set_title('NGC 2509')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
ax.legend()
plt.show()

# create training data for random forest
X = np.array(df[training_columns])
y = np.array(df['PMemb'] >= probability_threshold)
n_train = X.shape[0]
print(f'Training set size: {n_train}')

# create random forest
clf = RandomForestClassifier(max_depth=2, random_state=0)

# fit random forest to training data
clf.fit(X, y)

# print score
score = clf.score(X, y)
print(f'Random forest score on training data: {score}')
