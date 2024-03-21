import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('tweet1.csv')
df['Tweet'].value_counts()
train, test=train_test_split(df)
labelencoder=LabelEncoder()

train['Tweet'] = labelencoder.fit_transform(train['Tweet'])
test['Tweet'] = labelencoder.fit_transform(test['Tweet'])
print(train,test)