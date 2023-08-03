# Machine Learning for Predictive Pricing (Predicting House Prices)


## About The Project


**This project utilizes machine learning algorithms to predict house prices based on a pool of characteristics, 
ranging from the house square footage to the number of bathrooms to how the house was graded by the relevant 
authorities. It was originally completed as part of the final project for my course, 'Data Analysis with 
Python', offered online by IBM, however I expanded upon it to showcase more skills and techniques learnt 
through the course and make it wider in scope.**
<br>
<br>
**To build a machine learning model that can predict house prices, the house attributes that are most associated 
with price are identified, prepared and preprocessed, and finally used to train the model. More specifically, 
different models are developed with the data, trained, evaluated, and improved, before selecting the model that 
best accounts for the data, and therefore proves to be the best at producing valid and reliable price predictions. 
Each model is tested and verified through in-sample evaluation metrics to evaluate the model's performance in reference 
to the data fed to it, out-of-sample evaluations to estimate how the model is likely to perform in the real world, with 
novel datasets, and through visualizations to compare the distributions of the predicted prices to the actual prices in 
the dataset. Finally, the best model is selected and used to generate predictions.** <br>

<br>

**Overall, the project is broken down into five parts: <br>
&emsp; 1) Loading, Inspecting, and Cleaning the Data <br>
&emsp; 2) Data Preparation and Preprocessing <br>
&emsp; 3) Model Development and Evaluation <br>
&emsp; 4) Hyperparameter Tuning <br>
&emsp; 5) Model Prediction** <br>

<br>
<br>



## About The Dataset 
**The dataset being used here was taken from Kaggle.com, a popular website for finding and publishing datasets. 
You can quickly access it by clicking [here](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-wwwcourseraorg-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01). It presents house sales 
in Seattle-King County made between May 2014 to May 2015, and consisting of different house characteristics and 
the corresponding sale price for each house.** <br> 
<br>
**You can view each coloumn and its description in the table below:** <br>

| **Variable**      | **Description**                                                                                         |
| :-----------------| :------------------------------------------------------------------------------------------------------ |
| **id**            | Unique ID for each house sold                                                                           |
| **date**          | Date of the house sale                                                                                  |
| **price**         | Price of each house sold                                                                                |
| **bedrooms**      | Number of bedrooms                                                                                      |
| **bathrooms**     | Number of bathrooms                                                                                     |
| **sqft_living**   | Square footage of the house interior living space                                                       |
| **sqft_lot**      | Square footage of the lot (land space)                                                                  |
| **floors**        | Number of house floors                                                                                  |
| **waterfront**    | Whether a house is overlooking a waterfront (1) or not (0)                                              |
| **view**          | Rating of how good the house view is                                                                    |
| **condition**     | Rating of the overall house condition                                                                   |
| **grade**         | Overall grade given to the housing unit, based on King County grading system                            |
| **sqft_above**    | Square footage of the interior housing space that is above ground level                                 |
| **sqft_basement** | Square footage of the interior housing space that is below ground level                                 |
| **yr_built**      | Year the house was built                                                                                |
| **yr_renovated**  | Year when house was last renovated                                                                      |
| **zipcode**       | Zip code                                                                                                |
| **lat**           | Latitude coordinate                                                                                     |
| **long**          | Longitude coordinate                                                                                    |
| **sqft_living15** | Square footage of the interior housing living space for the closest 15 houses                           |
| **sqft_lot15**    | Square footage of the lot (land space) for the closest 15 houses                                        |


<br>
<br>

Here's a screenshot with a sample of the dataset:
<br> 

<img src="house sales screenshot.jpg" alt="https://github.com/Mo-Khalifa96/Machine-Learning-for-Predictive-Pricing/blob/main/house%20sales%20screenshot.jpg" width="800"/>

<br>
<br> 

## Aim 
**The aim of this project is to demonstrate my abilities and coding skills to build, evaluate, and deploy 
machine learning models for tasks such as predictive pricing.**
<br>
<br>

## Quick Access 
**For a quick access to the program, you can click on either of the links below. The first one renders the project 
ready for viewing whilst the second enables you to both view the code and also interact with it. Both links will 
direct you to a Jupyter notebook with the code and its resulting output, segregated and organized into separate blocks 
and provided with in-depth explanations or conclusions. If you wish to execute the code or reproduce the analysis results 
from the second link, make sure to run the first two cells, which will allow you to install and import all the Python packages 
that will be used across the project. To run any given block of code, you have to simply select the cell and click on the 'Run' 
icon on the notebook toolbar.**
<br>
<br>
<br>
***To view the project only, click on the following link:*** <br>
https://nbviewer.org/github/Mo-Khalifa96/Machine-Learning-for-Predictive-Pricing/blob/main/ML%20for%20Predictive%20Pricing%20%28Predictive%20House%20Prices%29%20-%20Jupyter%20version.ipynb
<br>
<br>
***Alternatively, to view the project and interact with its code, click on the following link:*** <br>
https://mybinder.org/v2/gh/Mo-Khalifa96/Machine-Learning-for-Predictive-Pricing/main?labpath=ML%20for%20Predictive%20Pricing%20(Predictive%20House%20Prices)%20-%20Jupyter%20version.ipynb
<br>
<br>

