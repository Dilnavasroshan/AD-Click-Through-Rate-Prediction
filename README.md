# AD-Click-Through-Rate-Prediction
Developing a predictive model for ad click-through rates (CTR). Uses data cleaning, visualization, correlation analysis, and machine learning classification. The Random Forest model is selected as the best model. Provides actionable insights for advertisers to optimize ad targeting and improve CTR.
#!/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import accuracy_score,confusion_matrix
import plotly.express as px

ad=pd.read_csv("ad_10000records.csv")
ad=ad.drop_duplicates()
ad["Gender"] = ad["Gender"].map({"Male": 1,"Female": 0})
ad=ad.drop(['City','Timestamp','Ad Topic Line'],axis=1)

print(ad.shape)
print(ad.info())

le=LabelEncoder()
ad["country"]=le.fit_transform(ad["Country"])
c1_map={index: label for index,label in enumerate(le.classes_)}
print(c1_map)
ad.drop(['Country'],axis=1,inplace=True)
print(ad.info())


counts=ad['country'].value_counts()
Country_m= counts.reset_index()
Country_m.columns= ['country', 'Count']

merged= pd.merge(ad,Country_m,on='country')
print(merged)
plt.figure(figsize=(8,6))

merged1 = merged.assign(Group_1=lambda x: x['Count'].apply(lambda val: 1 if val >= 300 else 0))
merged1= merged1.assign(Group_2=lambda x: x['Count'].apply(lambda val: 1 if val >= 200 and val < 300 else 0))
merged1 = merged1.assign(Group_3=lambda x: x['Count'].apply(lambda val: 1 if val >= 100 and val < 200 else 0))
merged1 = merged1.assign(Group_4=lambda x: x['Count'].apply(lambda val: 1 if val >= 0 and val < 100 else 0))



print(merged1)

sns.heatmap(merged1.corr(),annot=True,cmap='coolwarm',fmt="0.2f",annot_kws={"size":12})
plt.title('Correlation matrix')
plt.show()


fig = px.box(merged1, 
             x="Daily Time Spent on Site",  
             color="Clicked on Ad", 
             title="Click Through Rate based Time Spent on Site", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

fig = px.box(merged1, 
             x="Daily Internet Usage",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Daily Internet Usage", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

fig = px.box(merged1, 
             x="Age",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Age", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()






X=merged1.drop(['Clicked on Ad'], axis=1)
y=merged1['Clicked on Ad']


models=[]
models.append(('KNN',KNeighborsClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('RF',RandomForestClassifier()))
models.append(('SVM',svm.SVC()))

results=[]
names=[]
scoring='accuracy'
kfold=KFold(n_splits=10)
for name,model in models:
    cv_results=cross_val_score(model,X,y,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('cv results:',cv_results)
    print(f"accuracy of {name} is {cv_results.mean()}")

fig=plt.figure()
fig.suptitle("Algorithm Comparison")
ax=fig.add_subplot(111)
plt.boxplot(results)

ax.set_xticklabels(names)
plt.show()
