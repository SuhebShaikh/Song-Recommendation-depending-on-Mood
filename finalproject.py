import pandas as pd
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


le = preprocessing.LabelEncoder()

data=pd.read_csv('songs.csv')
print ("\n---------------------WELCOME TO SONG PREDICTION SYSTEM------------------------\n")
print ("\n<<<<<<<   Available songs    >>>>>>>\n")
print (data)

# Converting catagorical data to Numeric data 
df=data['Mood']
data['Mood']= le.fit_transform(df)
df1=data['Genre']
data['Genre']= le.fit_transform(df1)

# Binding in to new dataframe
test=data[['Genre','Mood','Popularity']]
test
test1= preprocessing.MinMaxScaler()
test11= test1.fit_transform(test)
n1= pd.DataFrame(test11)

#splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(test11,df, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Converting mood to numeric value
def Mood_to_numeric(y):
    if y=="Happy":
        return 0
    if y=="Horrific":
        return 1
    if y=="Love":
        return 2
    if y=="Peace":
        return 3	
    if y=="Sad":
        return 4
    if y=="Surprise":
        return 5
    if y=="Valor":
        return 6
      
def logistic():
	#Logistic Regression model  
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)
	
	print('\nAccuracy of Logistic regression classifier on training set: {:.2f} '.format(logreg.score(X_train, y_train)))
	print('\nAccuracy of Logistic regression classifier on test set: {:.2f} '.format(logreg.score(X_test, y_test)))
	y_pred = logreg.predict(X_test)
	print ("\n<<<<<   Predicted Mood     >>>>> \n") 

	y_pred1=list(set(y_pred))
	print (y_pred1)

	mood = raw_input("\n\nEnter Your mood :")
	print ("\n<<<<<<< Popular Songs For You >>>>>>>")

	i = y_pred1.index(mood.title())
	c1 = Mood_to_numeric(y_pred1[i])

	songs=data[['Song Title','Artist','Duration','Popularity']].loc[(data['Mood']==c1)]
	songs = songs.sort_values(by=['Popularity'],ascending=False)
	print(songs)

	songs1 = songs['Song Title']
	x_axis= le.fit_transform(songs1)
	pop = songs['Popularity']
	plt.bar(x_axis,pop,align='center',color=['violet', 'red', 'green', 'blue', 'cyan'],alpha=0.7)
	plt.xlabel('Song Title')
	plt.ylabel('Popularity')
	plt.ylim([0,100])
	plt.xticks(x_axis,songs1,fontsize=10,rotation=20)
	plt.title("Songs For your mood")
	plt.show()

	#print ("\n----Confusion Matrix-----\n\n")
	#print(confusion_matrix(y_test, y_pred))  
	print ("\n----Classification Report------\n\n")
	print(classification_report(y_test, y_pred))

def decisiontree():
	#Decision Tree Classifier model
	clf = DecisionTreeClassifier().fit(X_train, y_train)
	
	print('\nAccuracy of Decision Tree classifier on training set: {:.2f} '.format(clf.score(X_train, y_train)))
	print('\nAccuracy of Decision Tree classifier on test set: {:.2f} '.format(clf.score(X_test, y_test)))
	y_pred = clf.predict(X_test)
	print ("\n<<<<<   Predicted Mood     >>>>> \n")  

	y_pred1=list(set(y_pred))
	print (y_pred1)

	mood = raw_input("\n\nEnter Your mood :")
	print ("\n<<<<<<< Popular Songs For You >>>>>>>")

	i = y_pred1.index(mood.title())
	c1 = Mood_to_numeric(y_pred1[i])

	songs=data[['Song Title','Artist','Duration','Popularity']].loc[(data['Mood']==c1)]
	songs = songs.sort_values(by=['Popularity'],ascending=False)
	print(songs)

	songs1 = songs['Song Title']
	x_axis= le.fit_transform(songs1)
	pop = songs['Popularity']
	plt.bar(x_axis,pop,align='center',color=['violet', 'red', 'green', 'blue', 'cyan'],alpha=0.7)
	plt.xlabel('Song Title')
	plt.ylabel('Popularity')
	plt.ylim([0,100])
	plt.xticks(x_axis,songs1,fontsize=10,rotation=20)
	plt.title("Songs For your mood")
	plt.show()

	#print ("\n----Confusion Matrix-----\n\n")
	#print(confusion_matrix(y_test, y_pred)) 
	print ("\n----Classification Report------\n") 
	print(classification_report(y_test, y_pred))   
  
def knnmodel():
	#KNN classifier model
	knn = KNeighborsClassifier(n_neighbors=5)
	knn.fit(X_train, y_train)
	
	print('\nAccuracy of K-NN classifier on training set: {:.2f} '.format(knn.score(X_train, y_train)))
	print('\nAccuracy of K-NN classifier on test set: {:.2f} '.format(knn.score(X_test, y_test)))
	y_pred = knn.predict(X_test)
	print ("\n<<<<<   Predicted Mood     >>>>> \n") 

	y_pred1=list(set(y_pred))
	print (y_pred1)

	mood = raw_input("\n\nEnter Your mood :")
	print ("\n<<<<<<< Popular Songs For You >>>>>>>")

	i = y_pred1.index(mood.title())
	c1 = Mood_to_numeric(y_pred1[i])

	songs=data[['Song Title','Artist','Duration','Popularity']].loc[(data['Mood']==c1)]
	songs = songs.sort_values(by=['Popularity'],ascending=False)
	print(songs)
	
	songs1 = songs['Song Title']
	x_axis= le.fit_transform(songs1)
	pop = songs['Popularity']
	plt.bar(x_axis,pop,align='center',color=['violet', 'red', 'green', 'blue', 'cyan'],alpha=0.7)
	plt.xlabel('Song Title')
	plt.ylabel('Popularity')
	plt.ylim([0,100])
	plt.xticks(x_axis,songs1,fontsize=10,rotation=20)
	plt.title("Songs For your mood")
	plt.show()

	#print ("\n----Confusion Matrix-----\n\n")
	#print(confusion_matrix(y_test, y_pred))
	print ("\n----Classification Report------\n")  
	print(classification_report(y_test, y_pred)) 

def naive_bayes():
	#Gausian naive bayes Classifier model
	gnb = GaussianNB()
	gnb.fit(X_train, y_train)
	
	print('\nAccuracy of GNB classifier on training set: {:.2f} '.format(gnb.score(X_train, y_train)))
	print('\nAccuracy of GNB classifier on test set: {:.2f} '.format(gnb.score(X_test, y_test)))
	y_pred = gnb.predict(X_test)
	print ("\n<<<<<   Predicted Mood     >>>>> \n") 

	y_pred1=list(set(y_pred))
	print (y_pred1)

	mood = raw_input("\n\nEnter Your mood : ")
	print ("\n<<<<<<< Popular Songs For You >>>>>>>")

	i = y_pred1.index(mood.title())
	c1 = Mood_to_numeric(y_pred1[i])

	songs = data[['Song Title','Artist','Duration','Popularity']].loc[(data['Mood']==c1)]
	songs = songs.sort_values(by=['Popularity'],ascending=False)
	print(songs)
	
	songs1 = songs['Song Title']
	x_axis = le.fit_transform(songs1)
	pop = songs['Popularity']
	plt.bar(x_axis,pop,align='center',color=['violet', 'red', 'green', 'blue', 'cyan'],alpha=0.7)
	plt.xlabel('Song Title')
	plt.ylabel('Popularity')
	plt.ylim([0,100])
	plt.xticks(x_axis,songs1,fontsize=10,rotation=20)
	plt.title("Songs For your mood")
	plt.show()

	#print ("\n----Confusion Matrix-----\n\n")
	#print(confusion_matrix(y_test, y_pred))  
	print ("\n----Classification Report------\n")
	print(classification_report(y_test, y_pred)) 


check = 'y'
while check=='y' or check=='Y':
	print ("\n Available Models \n")
	model = raw_input("\n1. Logistic Regression Model \n2. Decision Tree Classifier Model \n3. K-nearest Neighbour classifier Model\n4. Naive Bayes classifier model\n\nEnter your choice : ")
	
	if model == '1' :
		logistic()
	elif model == '2' :
		decisiontree()
	elif model == '3' :
		knnmodel()
	elif model == '4' :
		naive_bayes()
	else :
		print ("Wrong Model selected")

	check = raw_input("\n\nCheck another Catagory (Y/N) :")

print ("Thank You........!")
