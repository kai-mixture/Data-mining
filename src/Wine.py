# import module
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#import dataset and split the dataset into training and testing set
wine_data_set = pd.read_csv("~/Documents/3年後期/データマイニング/レポート1/WINE/Data Set/winequality-white.csv",sep=";",header=0)

#count quality
count_data = wine_data_set.groupby('quality')['quality'].count()
print(count_data)

x = wine_data_set.drop(columns = 'quality')
y = wine_data_set['quality']

#Re-label
new_y = []

for d in y:
    if d <= 4:
        new_y.append(0)
    elif d <= 7:
        new_y.append(1)
    elif d <= 9:
        new_y.append(2)

#正規化
ms = MinMaxScaler()
x = ms.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,new_y,test_size=0.5, random_state=None)

#Prepare a classification model
clf = svm.LinearSVC()

#Learn training dataset
clf.fit(x_train, y_train)

#Predict with test dataset
pred = clf.predict(x_test)

#Check accuracy
print(accuracy_score(y_test, pred))



