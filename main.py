import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

insurance_data = pd.read_csv(r"D:/mona/mini_project/insurance.csv")
insurance_data.head()


# number of rows and columns
insurance_data.shape

# checking data points
print(insurance_data.size)

# getting some informations about the data
insurance_data.info()

# Elucidation of data set
insurance_data.describe()

# checking number of null value in the given data
insurance_data.isnull().sum()

# checking if any null value is present or not in the given data
insurance_data.isnull().any()

# checking value count of male and female in the given data
insurance_data['sex'].value_counts()

# Observation
# from the given data we can get the insights that :
# 1. data belongs to middle age people (mostly).
# 2. maximum age of person is 64 where as minimum age is 18.
# 3. maximum bmi is 53.13 which is a deep sign of obesity
# 4. there is no null value in the given data
# 5. There are 676 male and 662 female

# plotting a bar graph showing about number of male and female
insurance_data.sex.value_counts(normalize=False).plot.barh()
plt.show()

#pie chart: with Label and explode
# labels make a chart easier to understand because they show details about a data series or its individual data points
# To “explode” a pie chart means to make one of the wedges of the pie chart to stand out
mylables=["Male","Female"] # here label is "Male - is 1 where as Female - is 0"
colors = ['yellow', 'purple']
myexplode=[0.10,0]
size = [676, 662]
plt.pie(size,colors = colors,labels =mylables,explode = myexplode, shadow = True)
plt.title('PIE chart representing share of men and women in insurance data ')
plt.legend()
plt.show()

# checking customer belonging
insurance_data['region'].value_counts()

#plotting a Countplot showing region
sns.countplot("region",data = insurance_data)
plt.show()

#plotting a countplot showing age
plt.figure(figsize = (20,20))
sns.countplot("age",data = insurance_data)
plt.show()

# plotting a bar graph showing about region wise with labels grid and minor grids and title
x = ['North-East(0)', 'North-West(1)', 'South-East(2)', 'South-West(3)']
size = [324, 325, 364, 325]
plt.figure(figsize = (20,10))
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, size, color=['red', 'yellow', 'blue', 'green'])
plt.xlabel("Region")
plt.ylabel("Size")
plt.title("Number of Insurance Holder from All Over the region")
plt.xticks(x_pos, x)
# Turn on the grid
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
plt.show()

# 6.people belonging residential area from northeast(0) are 324 person ; northwest(1)

# checking children count
insurance_data['children'].value_counts()

#plotting a countplot showing number of children
plt.figure(figsize = (20,10))
sns.countplot("children",data = insurance_data)
plt.show()

#pie chart: with Label and explode
plt.figure(figsize = (20,10))
mylables=["No Children","1 Child","2 Children","3 Children","4 Children","5 Children"]
colors = ['green','pink','purple','yellow','red','blue']
myexplode=[0.10,0,0,0,0,0]
size = [574, 324,240,157,25,18]
plt.pie(size,colors = colors,labels =mylables,explode = myexplode, shadow = True)
plt.title('PIE chart representing share of person per chidren as per given data ')
plt.legend()
plt.show()

# checking number of smokers
insurance_data['smoker'].value_counts()

#plotting a bar grap showing number of smoker
plt.figure(figsize = (20,10))
mylables=['Non-Smoker','Non-Smoker']
colors = ['yellow','purple']
myexplode=[0.10,0.10]
size = [1064,274]
plt.pie(size,colors = colors,labels =mylables,explode = myexplode, shadow = True)
plt.show()

# 8. count of smokers are represented as 0 which is 1064 where as count of non-smokers are represented as 1 which is 274 

# pairplot
plt.figure(figsize = (30,30))
sns.pairplot(insurance_data)

# Corelation Between Diffrent Features
insurance_data[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']].corr()

#plot the correlation matrix of salary, balance and age in data dataframe.
plt.figure(figsize = (20,10))
sns.heatmap(insurance_data[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']].corr(), annot=True,cmap = "Greys")
plt.show()

insurance_data.columns

# Insigths From this Heat Map :-
# 1. Smoker Tends to Pay More Insurance Charges; 
# 2. Age is positively correlated with charges, with a correlation coefficient of 0.30. This suggests that as age increases, the cost of medical insurance tends to increase as well.
# 3.BMI is positively correlated with charges, with a correlation coefficient of 0.20. This suggests that as BMI increases, the cost of medical insurance tends to increase as well.
# 4.The number of children does not appear to have a strong correlation with charges, with a correlation coefficient of 0.07.
# 5.Sex and region also do not appear to have strong correlations with charges, with correlation coefficients of 0.06 and -0.01, respectively.

# Age vs Charges
# the more the age the more will be insurance charge (roughly estimated)
plt.figure(figsize = (18, 8))
sns.lineplot(x = 'age', y = 'charges', data = insurance_data)
plt.title("Age vs Charges")
plt.show()

#box plot for age vs charge
plt.figure(figsize = (30, 20))
sns.barplot(x = 'age', y = 'charges', data = insurance_data)
plt.title('age vs charges')
plt.show()

#plot the box plot of sex and charges
# as 1 belongs to men : it shows that men are paying more insurance charges then Women
#bar plot
plt.figure(figsize = (10, 5))
sns.barplot(x = 'sex', y = 'charges', data = insurance_data)
plt.title('sex vs charges')
plt.show()

# children vs charges
# no. of childrens of a person has a weird dependency on insurance charge. i.e(parents of more children tends to pay less insurance)
plt.figure(figsize = (10, 5))
sns.barplot(x = 'children', y = 'charges', data = insurance_data)
plt.title('CHILDREN VS CHARGES')
plt.show()

# region vs charges BAR GRAPh
plt.figure(figsize = (10, 5))
sns.barplot(x = 'region', y = 'charges', data = insurance_data, palette = 'colorblind')
plt.title('Region vs Charges')
plt.show()

# from the graph we can clearly state that region dont play any role in charges it is highly independent should be drop

# smoker vs charges
plt.figure(figsize = (10, 5))
sns.barplot(x = 'smoker', y = 'charges', data = insurance_data)
plt.title('SMOKERS VS CHARGES')
plt.show()

# BMI vs charges
plt.figure(figsize = (40,20))
sns.barplot(x = 'bmi', y = 'charges', data = insurance_data)
plt.title('BMI VS CHARGES')
plt.show()

# removing unrequired columns from the insurance data
# As from the above grph we can clearly state that region dont play any role in charge it is highly independent and should be dropped
insurance_data = insurance_data.drop('region', axis = 1)

insurance_data.shape

#as earlier there was 10704 data point the new one has 9366 data point after removing region
insurance_data.size

# seperate out features and target value from dataset
X=insurance_data.drop(["insuranceclaim"],axis=1).values
y=insurance_data["insuranceclaim"].values

X.shape

y.shape

#bmi outlier
sns.boxplot(insurance_data["bmi"])
plt.show()

# Finding Position of Outlier
#position plot of outlier
print(np.where(insurance_data["bmi"]>45))

#Children outlier
sns.boxplot(insurance_data["children"])
plt.show()

#Charges outlier
sns.boxplot(insurance_data["charges"])
plt.show()


# Charges can be More or less as per required by insurance company

#spliting data into training and testing data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.27, random_state =4)

print("X_train shape : " , X_train.shape)
print("X_test shape : ", X_test.shape)
print("y_train shape : " , y_train.shape)
print("y_test shape : ", y_test.shape)

from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.linear_model import LogisticRegression 

# Logistics Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion matrix: \n", conf_matrix)
print("Where ; True Positive: is 104; False Positive: is 40; True Negative : is 24; False Negative is 189")

# compute accuracy on training set 
logreg_train= logreg.score(X_train,y_train)
print("Training Data Accuracy by Logistics Regression Algorithm is : " ,logreg_train)
# compute accuracy on testing set
logreg_test= logreg.score(X_test,y_test)
print("Testing Data Accuracy by Logistics Regression is : " , logreg_test)

# Evaluate the model on the test data
score = logreg.score(X_test, y_test)
print("Accuracy of Logistic Regression is : ",score)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2, axis = None)
print("MSE :", mse)
# Calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

from sklearn.ensemble import RandomForestClassifier

# model = RF Random Forest
rf = RandomForestClassifier(n_estimators=1000,random_state=45)

#fitting model
rf.fit(X_train,y_train)

#predicting

y_pred=rf.predict(X_test)

y_pred

# compute accuracy on training set

rf_train= rf.score(X_train,y_train)
print("Training Data Accuracy by Random Forest Algorithm is : " , rf_train)
# compute accuracy on testing set
rf_test= rf.score(X_test,y_test)
print("Testing Data Accuracy by Random Forest Algorithm is : " , rf_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2, axis = None)
print("MSE :", mse)

# Calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

from sklearn.tree import DecisionTreeClassifier

# model
dtc = DecisionTreeClassifier()
#fitting
dtc.fit(X_train,y_train)

#predicting via Decision Tree Algorithm
y_pred=dtc.predict(X_test)
y_pred

#Calculating RMSE Root Mean Square Error
rmse= np.sqrt(metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Square Error = ",rmse)

y_pred_df=pd.DataFrame(y_pred)
y_pred_df

y_pred_df["Actual"]=y_test
y_pred_df


y_pred_df.columns=["Predicated","Actual"]
y_pred_df

# compute accuracy on training set
dtc_train= dtc.score(X_train,y_train)
print("Training Data Accuracy by Decision Tree Algorithm is : " , dtc_train)
# compute accuracy on testing set
dtc_test= dtc.score(X_test,y_test)
print("Testing Data Accuracy by Decision Tree Algorithm is : " , dtc_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2, axis = None)
print("MSE :", mse)

# Calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

from sklearn.naive_bayes import GaussianNB

#model naive Bayes
nb = GaussianNB()
nb.fit(X_train,y_train)

print("Naive Bayes Score : ",nb.score(X_test,y_test))


#prediction
y_pred= nb.predict(X_test)
print(y_pred)

y_pred.size

# compute accuracy on training set
nb_train= nb.score(X_train,y_train)
print("Training Data Accuracy by Random Forest Algorithm is : " ,nb_train)
# compute accuracy on testing set
nb_test= nb.score(X_test,y_test)
print("Testing Data Accuracy by Random Forest Algorithm is : " ,nb_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2, axis = None)
print("MSE :", mse)
# Calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

# Genrally we select only those model which has highest accuracy among all the prediction result

if dtc_test > rf_test:
 print (" For this Data Highest Accuracy belong to Decision Tree, out of 4 model ")
elif rf_test > nb_test:
 print (" For this Highest Accuracy belong Data Random Forest, out of 4 model ")
elif nb_test> logreg_test:
 print(" For this Highest Accuracy belong Data Naive Bayes classifier, out of 4 model ")
elif logreg_test>dtc_test:
 print(" For this Highest Accuracy belong Data Logistics Regression, out of 4 model ")