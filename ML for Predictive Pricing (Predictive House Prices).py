#MACHINE LEARNING FOR PREDICTIVE PRICING: PREDICTING HOUSE PRICES


from sklearnex import patch_sklearn
patch_sklearn()
import warnings
warnings.simplefilter("ignore")

#Importing the Python modules to be used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


#Part One: Loading, Inspecting, and Cleaning the Data
#1. Loading and reading the dataset  
#Accessing the file 
df = pd.read_excel("House Sales in King County.xlsx")

#Previewing the first 10 entries off the dataset 
print(df.head(10))
print('')


#2. Inspecting the Data
#Inspecting the shape (rows x coloumns) of the dataframe
shape = df.shape
print('Number of coloumns:', shape[1])
print('Number of rows:', shape[0])
print('')


#Inspecting the coloumn headers, data type, and entries count 
print(df.info())
print('')


#Get statistical summary of the dataset 
print(df.describe())
print('')


#Show distribution of values across the dataset 
df.hist(figsize=(20, 12))
print('')


#3. Cleaning Up and Updating the Data 
#Reporting total sum of empty/NaN (not-a-number) values for each coloumn
print('Number of empty/NaN entries per coloumn:') 
print(df.isnull().sum())
print('')


#Now replacing null entries with the mean value for the necessary coloumns
#calculate mean value for coloumn 'bedrooms'
mean_bedrooms = df['bedrooms'].mean()
#replacing null entries with the mean value 
df['bedrooms'].replace(np.nan, mean_bedrooms, inplace=True)

#calculate mean value for coloumn 'bathrooms'
mean_bathrooms = df['bathrooms'].mean()
#replacing null entries with mean value 
df['bathrooms'].replace(np.nan, mean_bathrooms, inplace=True)


#Previewing the coloumns that had null values again 
print("Number of null values for the column \'bedrooms\':", df['bedrooms'].isnull().sum())
print("Number of null values for the column \'bathrooms\':", df['bathrooms'].isnull().sum())
print('\n\n')



#Part Two: Data Preparation and Preprocessing
#1. Identifying the Variables  
#Performing correlational analysis to identify and select the attributes that  
# are most correlated with house price.

#Checking the correlations between all variables in the dataset
correlations_table = df.corr()
print(correlations_table)
print('')


#Showing the correlations with house price only (from highest to lowest)
correlations_ByPrice = df.corr()['price'].sort_values(ascending=False)
print(correlations_ByPrice)
print('\n\n')


#Now selecting the variables for training the model 
#specifying the independent/predictor variables
predictors = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 
        'sqft_basement', 'bedrooms', 'lat', 'waterfront']

#assigning the variables to 'x_data'
x_data = df[predictors]

#specifying the dependent/target variable and assigning it to 'y_data'
y_data = df['price']



#2. Data Splitting 
#Performing data splitting to obtain a training set (75%) and testing set (25%)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.75, random_state=0)

#We can check the sizes of the training and testing sets
print('Number of training samples:', x_train.shape[0])    
print('Number of testing samples:', x_test.shape[0])
print('')


#3. Feature Scaling: Standardizing the Scales  
#Standardizing the scales so that the data would have the properties of a standard normal 
# distribution, with a mean of 0 and standard deviation of 1.

#Get a scaler object 
Scaler = StandardScaler()
#scaling the training set 
x_train = Scaler.fit_transform(x_train)  
#scaling the testing set 
x_test = Scaler.transform(x_test)



#Part Three: Model Development and Evaluation 

#MODEL ONE: Multiple Linear Regression Model 
#Creating a regression object 
multireg_model = LinearRegression()

#Training the model with training data (i.e. fitting the model)
multireg_model.fit(x_train, y_train)

#Evaluating the model with the testing set using R-squared 
R2_test = multireg_model.score(x_test, y_test)
print(f'The R-squared score for the multiple regression model is: r2={round(R2_test,3)}')
print('')


#Now that we have fitted the model, we can generate price predictions 
#Generating predictions using the testing set
Y_pred = multireg_model.predict(x_test) 

#We can compare the actual prices vs. predicted prices 
Actual_vs_Predicted = pd.concat([pd.Series(y_test.values), pd.Series(Y_pred)], axis=1, ignore_index=True).rename(columns={0:'Actual Prices', 1:'Predicted Prices'})
Actual_vs_Predicted['Actual Prices'] = Actual_vs_Predicted['Actual Prices'].apply(lambda price: '${:,.2f}'.format(price))
Actual_vs_Predicted['Predicted Prices'] = Actual_vs_Predicted['Predicted Prices'].apply(lambda price: '${:,.2f}'.format(price))

#Previewing the first 10 price comparisons 
print(Actual_vs_Predicted.head(10))
print('')


#Model Evaluation: Root Mean Squared Error 
#First, calculating the mean squared error (MSE)
MSE = mean_squared_error(y_test, Y_pred)

#Calculating the root MSE (RMSE)
RMSE = np.sqrt(MSE)
print(f'The root mean squared error is: RMSE={round(RMSE,3)}')
print('')


#Model Evaluation: Distribution plot 
#Visualizing the discrepancy between the actual prices and the predicted prices using a distribution 
# plot (based on kernel density estimation) to get better insight and understanding of the model  

#Visualizing the distribution of actual vs. predicted prices 
#Creating the distribution plot 
ax1 = sns.distplot(y_test, hist=False, label='Actual Values')
sns.distplot(Y_pred, ax=ax1, hist=False, label='Predicted Values')
#Adding a title and labeling the axes
plt.title('Actual vs. Predicted Values for house Prices')
plt.xlabel('House Price (in USD)', fontsize=12)
plt.ylabel('Distribution density of price values', fontsize=12)
plt.legend(loc='best')
#Adjusting the x-axis to display the prices in a reader-friendly format
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.xticks(rotation=90)

#Displaying the distribution plot
plt.show()




#MODEL TWO: Multivariate Polynomial Regression Model 
#Performing cross validation to identify the best polynomial order for the model 
#First, specifying the polynomial orders to test out
poly_orders = [2,3,4,5]       #up to five polynomials 

#Now testing out different orders using cross validation to select the best 
cv_scores = {}
for order in poly_orders: 
    #creating polynomial features object
    poly_features = PolynomialFeatures(degree=order)
    #transforming predictor variables to polynomial features (for both sets)
    x_train_poly = poly_features.fit_transform(x_train)
    x_test_poly = poly_features.fit_transform(x_test)

    #creating a regression object
    polyreg_model = LinearRegression()

    #Now using 10-fold cross validation to determine the best polynomial order 
    r2_scores = cross_val_score(polyreg_model, x_train_poly, y_train, cv=10)
    
    #Retrieving mean R-squared score for a given polynomial order 
    cv_scores[order] = np.mean(r2_scores)
    

#Selecting the best polynomial order 
best_order, best_score = None, None  
for order,score in cv_scores.items():
    if best_score is None or best_score < score: 
        if score > 0:
            best_score = score 
            best_order = order 

#Reporting the best model with the most optimal polynomial 
print(f'The best model for the data has a polynomial order of {best_order}, and R-squared score of: r2={round(best_score,3)}')
print('')


#Model Testing 
#Testing the model again using the testing set to get the best estimate of its 
# performance in the real world.

#Creating a polynomial features object
poly_features = PolynomialFeatures(degree=best_order)
#transforming predictor variables to polynomial features
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.fit_transform(x_test)
#fitting the model 
polyreg_model = LinearRegression()
polyreg_model.fit(x_train_poly, y_train)

#Testing the model with the test set (using r-squared)
R2_test = polyreg_model.score(x_test_poly, y_test) 
print(f'The r-squared score for the testing set is: r2={round(R2_test,3)}')
print('')


#Model Evaluation: Root Mean Squared Error
#Generating predictions using both sets 
Y_pred_train = polyreg_model.predict(x_train_poly)
Y_pred_test = polyreg_model.predict(x_test_poly)
#Calculating root mean squared error for both sets to compare them
MSE_train = mean_squared_error(y_train, Y_pred_train) 
RMSE_train = np.sqrt(MSE_train)
MSE_test = mean_squared_error(y_test, Y_pred_test)
RMSE_test = np.sqrt(MSE_test)
print('RMSE for training set: {:,.3f}'.format(RMSE_train))
print('RMSE for testing set: {:,.3f}'.format(RMSE_test))
print('')


#Model Evaluation: Distribution Plot
#Setting the characteristics of the plots 
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(10, 5))
#Visualizing model fitting for the training set 
ax1 = sns.distplot(y_train, hist=False, ax=axes[0], label='Actual Values (Training)')
sns.distplot(Y_pred_train, hist=False, ax=ax1, label='Predicted Values (Training)')
#Visualizing model fitting for testing set 
ax2 = sns.distplot(y_test, hist=False, ax=axes[1], label='Actual Values (Testing)')
sns.distplot(Y_pred_test, hist=False, ax=ax2, label='Predicted Values (Testing')

#Adding titles and labeling the axes 
fig.suptitle('Model performance in-sample vs. out-of-sample')
axes[0].set_title('Model fitting with training set')
axes[0].set_xlabel('House Price (in USD)')
axes[0].set_ylabel('Distribution density of price values')
axes[0].legend(loc='best')
axes[1].set_title('Model fitting with testing set')
axes[1].set_xlabel('House Price (in USD)')
axes[1].set_ylabel('Distribution density of price values')
axes[1].legend(loc='best')
#Adjusting the x-axis to display the prices in a reader-friendly format
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.gcf().axes[1].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))

#show plot
plt.show()



#Part Four: Hyperparameter Tuning 
#Performing L1 regularization using lasso regression to improve model prediction and avoid 
# multicollinearity. We can take a second look at the correlations between the current 
# predictors using a heatmap to identify potential multicollinearity in the data.

#Plotting a heatmap 
#Specifying the figure size 
plt.figure(figsize=(8,4))     
mask = np.triu(np.ones_like(df[predictors].corr(), dtype=bool))
sns.heatmap(df[predictors].corr(), annot=True, mask=mask, cmap='Blues', vmin=-1, vmax=1)
plt.title('Correlation Coefficient Of Predictors', fontsize=14)
plt.show()


#MODEL DEVELOPMENT: Polynomial Lasso Regression Model 
#Creating a pipeline to automate model development 
#Specifying the steps 
pipe_steps = [('Polynomial', PolynomialFeatures()),    #to perform a polynomial transform
                ('Model', Lasso())]      #to develop the lasso regression model      


#Creating the pipeline to build a lasso regression model with polynomial features 
lasso_model = Pipeline(pipe_steps)


#Grid Search 
#Now performing grid search to obtain the best polynomial order and alpha value
#specifying the hyperparameters to test out 
parameters = {'Polynomial__degree': [2,3,4,5],      #specifying the polynomials to test out
              'Model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}     #specifying the alpha values to test out

#Creating a grid object and specifying the cross-validation characteristics
Grid = GridSearchCV(lasso_model, parameters, scoring='r2', cv=10) 

#Fitting the model with the training data for cross validation 
Grid.fit(x_train, y_train)


#Reporting the results of the cross validation (best polynomial order, alpha, and r2 score)
best_order = Grid.best_params_['Polynomial__degree']
best_alpha = Grid.best_params_['Model__alpha']
best_r2 = Grid.best_score_
print(f'The best model has a polynomial order of {best_order}, alpha value of: alpha={best_alpha}, and r-squared score of r2={round(best_r2,3)}')
print('')


#Model Testing 
#Testing the model one final time using the testing set 
#First, extracting the model with the best parameters 
Lasso_Model = Grid.best_estimator_

#Calculating the R-squared score for the model using the testing set 
R2_test = Lasso_Model.score(x_test, y_test)
print(f'The r-squared score for the testing set is: r2={round(R2_test,3)}')
print('')


#Model Evaluation: Root Mean Squared Error  
#Generating predictions using both sets 
Y_pred_train = Lasso_Model.predict(x_train)
Y_pred_test = Lasso_Model.predict(x_test)
#Calculating root mean squared error for both sets to compare them
MSE_train = mean_squared_error(y_train, Y_pred_train) 
RMSE_train = np.sqrt(MSE_train)
MSE_test = mean_squared_error(y_test, Y_pred_test)
RMSE_test = np.sqrt(MSE_test)
print('RMSE for training set: {:,.3f}'.format(RMSE_train))
print('RMSE for testing set: {:,.3f}'.format(RMSE_test))
print('')


#Model Evaluation: Distribution Plot 
#Setting the characteristics of the plots 
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 5))
#Visualizing model fitting for the training set 
ax1 = sns.distplot(y_train, hist=False, ax=axes[0], label='Actual Values (Training)')
sns.distplot(Y_pred_train, hist=False, ax=ax1, label='Predicted Values (Training)')
#Visualizing model fitting for testing set 
ax2 = sns.distplot(y_test, hist=False, ax=axes[1], label='Actual Values (Testing)')
sns.distplot(Y_pred_test, hist=False, ax=ax2, label='Predicted Values (Testing')

#Adding titles and labeling the axes 
fig.suptitle('Model performance in-sample vs. out-of-sample')
axes[0].set_title('Lasso Model fitting with training set')
axes[0].set_xlabel('House Price (in USD)')
axes[0].set_ylabel('Distribution density of price values')
axes[0].legend(loc='best')
axes[1].set_title('Lasso Model fitting with testing set')
axes[1].set_xlabel('House Price (in USD)')
axes[1].set_ylabel('Distribution density of price values')
axes[1].legend(loc='best')
#Adjusting the x-axis to display the prices in a reader-friendly format
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
plt.gcf().axes[0].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))
plt.gcf().axes[1].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('${x:,.0f}'))

#show plot
plt.show()



#Part Five: Model Prediction  
#For this section, I will create a custom function that takes a set of data comprised of different house 
# characteristics, and based on it the final model will be used to produce a price prediction that best 
# suits these particular characteristics. 

#Creating a pipeline to automate model development (includes feature scaling, 
# polynomial transform, and lasso regression)
#Specifying the pipeline process 
pipeline_steps = [('Scaler', StandardScaler()), 
                 ('Polynomial', PolynomialFeatures(degree=3)),
                 ('Model', Lasso(alpha=10000))]

#Building the pipeline for the lasso regression model 
Model = Pipeline(pipeline_steps)

#Fitting the model with the entire dataset 
Model.fit(x_data, y_data)


#Now creating a custom function, MakePredictions(), that takes novel data of different house characteristics
# and uses the model to generate predictions that best suit the new characteristics passed to the function. 
#Defining the function 
def MakePrediction(model, X_vars): 
    """This function takes two inputs: 'model', which specifies the model to be used to generate the price predictions, 
    and 'X_vars', which specifies the house characteristics for each house to make the price prediction based on. It 
    runs the prediction making process and returns a table with the predicted prices for each house."""

    Y_pred = model.predict(X_vars)
    Y_pred_df = pd.Series(Y_pred, name='Predicted Prices').to_frame().apply(lambda series: series.apply(lambda price: '${:,.2f}'.format(price)))
    return Y_pred_df 


#To test the function I will extract a random sample from the original dataset for the lack  
# there's no new data available. 
#Extracting a random sample and assigning it to 'X_new'
X_new = x_data.sample(20)        #number of samples = 20  

#Previewing the sample 
print(X_new)
print('')

#Now using the custom function to generate price predictions 
# from the sample, X_new 
print(MakePrediction(Model, X_new)) 
print('')

#Showing the house characteristics and the corresponding predicted prices together
sample_and_prediction, sample_and_prediction['Predicted Prices'] = X_new.reset_index(drop=True), MakePrediction(Model, X_new)
print(sample_and_prediction)


#END
