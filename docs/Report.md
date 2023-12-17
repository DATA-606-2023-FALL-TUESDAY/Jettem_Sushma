##				REPORT

## 1. Project Title:  Addressing the Opioid Epidemic: A National Public Health Challenge 
#### Domain: Government / Health

- **Author's Name:** Sushma Jettem
- **Prepared for:**  UMBC Data Science Master's Degree Capstone by Dr Chaojie (Jay) Wang
- **Semester:** Fall 2023

- <a href="https://www.youtube.com/watch?v=sPD5mj7OKy8"><img align="left" src="https://img.shields.io/badge/-YouTube Presentation-FF0000?logo=youtube&style=flat" alt="icon | YouTube"/></a> 


- <a href="https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/blob/main/docs/Final_Presentation.pptx"><img src="https://img.shields.io/badge/-PowerPoint Presentation Download-B7472A?logo=microsoftpowerpoint&style=flat" alt="icon | GitHub"/></a>  
 
- <a href="https://github.com/sushmajettem05"><img align="left" src="https://img.shields.io/badge/-GitHub-181717?logo=github&style=flat" alt="icon | GitHub"/></a>


  

## 2. Background: Tackling the Opioid Crisis in the United States and San Francisco

In the United States, drug overdoses have become the primary cause of unintentional deaths, with opioids driving a significant portion of these fatalities. The historical "War on Drugs" initiative, initiated in the 1970s, has evolved over time, contributing to an opioid crisis across several states. San Francisco, known for pioneering progressive health solutions, is currently facing a substantial drug issue and opioid crisis.

To combat this, the city has embraced innovative approaches, including Safe Injection Sites (SIS). SIS provides supervised spaces for individuals to consume intravenous drugs, aiming to reduce public disturbances associated with drug use. San Francisco's commitment to harm reduction aligns with its history of implementing forward-thinking public health measures. As we delve into this paper, we'll explore the origins of the opioid crisis, the impact of the "War on Drugs," and the novel strategies, such as Safe Injection Sites, implemented by San Francisco to address this pressing public health challenge.

## Research questions: 
- Analyzing various crime categories across diverse neighborhoods: What are the top 5 neighborhoods with the highest rates of assaults? Are there specific combinations of crimes that commonly occur together in particular areas?
- Identifying potential locations for implementing Safe Injection Sites (SIS) on behalf of the San Francisco government.
- Do specific demographic groups or geographical areas experience a higher incidence of fatal opioid overdoses? 
- If such disparities exist, is there a discernible connection between prescription practices and opioid-related deaths within these demographics or regions?


## 3. Data:
- **Data sources/tools:** 1. https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry/data
2) https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/data

The San Francisco Police Department's database serves as the repository for collecting crime data. This historical dataset spans from January 2003 to May 2018. 

**Data Size:** - 219.7 MB

**Data shape:** 
  - Rows = 2,215,024
  - Columns = 13 

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/34253b2a-7d46-447a-a533-b49611b60108)
 
**Data Info:**
Below we can see that Variables and its types Associated with the data and data size.
 

Below we can see Different Category of Crimes and its Reports.
 

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/59cc821b-1d41-4206-9120-193a34d2e441)


**Let’s Focus on Drugs:**
 

Types of Districts associated with Drugs:
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/82d19b1c-8052-4b6d-bb2d-5cb212d1deda)


- **Target Variable(s)** - Category, Drug/Narcotic.

## 4. Analysis Approach and Inferences:
1. For every type of crime, we tallied the incidents and created a plot. The distribution was skewed, so we took the log to normalise it. The normalised distribution of crime categories is shown below. 

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/94865780-ed5d-4a2d-b49d-9656d4e49e97)


2. There were 915 different criminal descriptors; these descriptions help identify whether a crime involves drugs or not. In order to create the cluster maps, we tallied the occurrences for each crime description, filtered out those that fell below the 97th percentile, and retained the remaining ones.
3. To investigate the distributions of various categories (i.e., crimes throughout each PdDistrict, or Police District), a cluster map was created. Once more, the skewed distribution had an impact on our model, which is displayed below.Grand Theft Auto is an outlier, as can be seen, and we don't get any other information, therefore normalisation was necessary.
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/f9dc3e97-9671-4247-aa30-667ae0f6a3dc)


4. Therefore, min-max normalisation was used for normalisation since logarithmic normalisation loses the scale that indicates how big or tiny a feature is in relation to another. The normalised cluster map is shown below.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/d5c99199-0f66-4f55-86b8-d8d132a2a428)

 
 **Here we can observe the following:**
-  Southern: extremely high occurrences of theft, including theft from auto
-  Bayview: significant occurrences of violences and threats
-  Tenderloin: appears to be an anomaly, with incredibly high rates of drug paraphernalia possession. Though it might be a false positive (i.e., these could be related to marijuana), tenderloin appears to be a good candidate for SIS installation. Therefore, more research is required.

5. Subsequently, we employed regular expressions and string pattern matching to filter crimes linked to drugs, and we tallied the instances of each unique description of drugs. The cluster map below was impacted by the distribution's skew once more. Tenderloin is an anomaly, as can be seen, and we learn nothing more..
  
 ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/858bbc2f-a486-420e-992c-0e974c049106)

 
6. Below is normalized cluster-map which shows distribution of narcotics related crimes across each PdDistrict.
 ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/365fe16e-bee8-4147-831a-3f186b51215c)

Inference:From the cluster-map above we can clearly conclude that, Tenderloin, Southern, Mission and Northern are the optimal candidates for installing SIS

7. Time-series analysis was then carried out to examine opioid patterns over time. Initially, we created opioid categories and features by compressing a large number of criminal descriptions pertaining to drugs (such as barbiturate, coke, marijuana, meth, etc.). Next, we made a 30-day frame for every group, and from January 1, 2003, to May 15, 2018, we counted the number of occurrences for every group during this 30-day window, or every month. In order to eliminate monthly cyclical patterns, we indexed the months from 0 to 187. To illustrate these trends, a stacked histogram is shown below. As you can see, there was a notable increase in instances involving meth and heroin.
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/4fb42740-bf97-44ba-9474-4645950801d6)


8. To make the trends clearer, below is the normalized distribution of opioid trends across the years from 2003 to 2018.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/96bb820f-0d21-43e3-9149-f6d24a5c6006)

 
It is evident that over this time, the number of crack-related incidents decreased. In a similar vein, occurrences using marijuana decreased following its legalization in 2016. However, crimes involving meth and heroin have increased dramatically; this is a strong indication that the problem is widespread.


## 5. EDA

- **Chart 1:** Count plot of incidents per year
This chart shows the number of incidents per year from 2016 to 2020. The chart shows that the number of incidents has been increasing steadily over the years, with a sharp increase in 2020.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/8b6b3241-09ec-45c5-b129-74597072f7dc)

- **Chart 2:** Day-wise distribution of incidents
This chart shows the day-wise distribution of incidents from 2016 to 2020. The chart shows that incidents are more likely to occur on weekdays than on weekends.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/5b027b47-905b-4f12-a163-632783e8bfee)

- **Chart 3:** Monthly trends over the years
This chart shows the monthly trend of incidents from 2016 to 2020. The chart shows that incidents are more likely to occur in the summer months than in the winter months.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/9336a052-b704-4b97-ada8-ad0c3e98e4ca)

- **Chart 4:** Top 10 districts with the highest number of incidents
This chart shows the top 10 districts with the highest number of incidents from 2016 to 2020. The chart shows that the district with the highest number of incidents is located in the southern part of the country.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/b1163fe9-e4bf-4c05-92c4-0708a604835e)


**Overall insights from the charts:**
-The number of incidents has been increasing steadily over the years, with a sharp increase in 2020.
-Incidents are more likely to occur on weekdays than on weekends.
-Incidents are more likely to occur in the summer months than in the winter months.
-The district with the highest number of incidents is located in the southern part of the country.

**Interpretations of the charts:**
- The increase in the number of incidents over the years may be due to a number of factors, such as increased access to drugs, changes in drug use patterns, and improved reporting of incidents.
- The fact that incidents are more likely to occur on weekdays and in the summer months may suggest that drug use is more common among people who are employed and/or who have more leisure time.
- The fact that the district with the highest number of incidents is located in the southern part of the country may be due to several factors, such as poverty, unemployment, and lack of access to treatment.


**Chart 1:** Incident resolution types
This chart shows the percentage of incidents that were resolved using different methods. The chart shows that the most common method of incident resolution was self-service, followed by analyst resolution and escalations.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/d175f9c3-4a8f-4a9e-938c-a167e2b59fc9)

**Chart 2:** Heatmap showing monthly incident counts over the years.
This chart shows the number of incidents that occurred in each month from 2016 to 2023. The chart shows that incidents are most likely to occur in the summer months and least likely to occur in the winter months.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/73e31845-cb6a-49c2-b80b-0f8e8e320171)

**Chart 3:** Duration between reported and occurred dates.
This chart shows the distribution of the time between when an incident was reported and when it occurred. The chart shows that most incidents are reported within a few hours of occurring.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/3922f853-9a9a-465d-bb54-434fb5e37164)


**Overall insights from the charts:**
- The majority of incidents are resolved using self-service methods.
- Incidents are most likely to occur in the summer months and least likely to occur in the winter months.
- Most incidents are reported within a few hours of occurring.

**Interpretations of the charts:**
•	The fact that the majority of incidents are resolved using self-service methods suggests that users can find the information and resources they need to resolve their issues on their own.
•	The fact that incidents are most likely to occur in the summer months may be due to several factors, such as increased usage of IT systems during the summer months and changes in user behavior during the summer months.
•	The fact that most incidents are reported within a few hours of occurring suggests that users are aware of the importance of reporting incidents promptly.

**Pie chart of incident categories**

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/482df460-6c04-4f0a-ac7a-117fe2637af2)

- i.	Possession of base/rock cocaine is the most common drug-related incident, accounting for 24.8% of all incidents.
- ii.	Possession of narcotics paraphernalia is the second most common incident, accounting for 15.5% of all incidents.
- iii.	Possession of marijuana is the third most common incident, accounting for 12.3% of all incidents.
- iv.	Possession of cocaine, possession of controlled substances, and sale of base/rock cocaine are all tied for the fourth most common incident, each accounting for 9.8% of all incidents.
- v.	Possession of heroin, possession of methamphetamine, possession of marijuana for sale, and possession of base/rock cocaine for sale are all less common incidents, each accounting for less than 6% of all incidents.


## 6. Data Preprocessing:
- Handled NA Values:
    - Mitigated missing or null values within the dataset through deletion or imputation strategies.
- Handled Categorical and Objects:
    - Processed categorical variables or object types to make them usable for modeling by employing encoding techniques.
- Scaled the Data:
    - Applied scaling methods to normalize or standardize numerical data, ensuring uniformity in their scales.
- Time Variable Conversions:
    - Transformed time-based variables into appropriate formats (e.g., converting strings to datetime objects) for analysis.
- Feature Selections:
    - Utilized methodologies to identify and choose relevant features, optimizing the model's performance and reducing complexity.
- One Hot Encoding:
    - Transformed categorical variables into binary vectors (0s and 1s) to represent multiple categories in machine learning models.
- Label Encoding:
    - Converted categorical labels into numerical form to facilitate model training by assigning unique integers to different categories.
- Created New Feature:
    - Engineered additional features based on existing data or domain knowledge to enhance model predictive capability.
- Train Test Split:
    - Segregated the dataset into training and testing subsets to evaluate model performance on unseen data accurately.


## 7. Models to Be trained:
**Logistic Regression**
Logistic regression is a statistical model that predicts the probability of a binary outcome (e.g., yes or no, pass or fail, spam or not spam). It is a linear model that uses a logistic function to map the input features to a probability between 0 and 1.
Logistic regression is a popular choice for binary classification tasks because it is simple to understand and interpret. It is also relatively robust to outliers and missing data.
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/60a38127-298b-4588-9734-13b54fc92cc2)


**K-Nearest Neighbors (KNN)**
KNN is a non-parametric algorithm that classifies new data points based on the majority class of their nearest neighbors in the training data. The number of nearest neighbors is specified by the parameter k.
KNN is a simple and versatile algorithm that can be used for both classification and regression tasks. It is also non-parametric, which means that it does not make any assumptions about the underlying data distribution.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/eb399498-0eb9-4616-a31c-06e948c64def)


**Random Forest**
Random forest is an ensemble method that combines multiple decision trees to make predictions. Each decision tree is trained on a random subset of the training data and a random subset of the features. The predictions of the individual decision trees are then aggregated to produce the final prediction.
Random forest is a powerful algorithm that can handle both linear and nonlinear relationships. It is also relatively robust to outliers and noise.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/1dbc5e36-3156-46c0-8522-015ef61191e1)


**Xgboost**
Xgboost is an ensemble method that is based on boosting. Boosting is an iterative process that trains a sequence of models, each of which is designed to improve on the predictions of the previous model. Xgboost uses a gradient boosting framework, which means that each model is trained to minimize the gradient of the loss function concerning the model parameters.
Xgboost is a state-of-the-art algorithm that can achieve high accuracy on a wide range of tasks. It is also relatively efficient to train and predict.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/52255f2f-93c6-4021-bb22-8bc51e332c7e)

| Feature                          | Logistic Regression      | KNN                                | Random Forest                     | Xgboost                            |
|----------------------------------|--------------------------|------------------------------------|-----------------------------------|------------------------------------|
| Model type                       | Parametric               | Non-parametric                     | Ensemble                          | Ensemble                           |
| Classification or regression     | Classification          | Classification and regression      | Classification and regression     | Classification and regression      |
| Assumptions about data distribution | Linear relationships  | No assumptions                    | No assumptions                    | No assumptions                     |
| Robustness to outliers and noise  | Relatively robust        | Sensitive to outliers              | Relatively robust                 | Relatively robust                  |
| Interpretability                 | High                     | Low                                | Low                               | Low                                |
| Efficiency                       | Efficient                | Efficient for small datasets, less efficient for large datasets | Less efficient                   | Less efficient                    |

 **Data Imbalance**
Data imbalance occurs when a dataset has an unequal distribution of data points across different classes or categories. For instance, in a dataset of emails, if there are far more spam emails than not spam emails, the dataset is considered imbalanced. This imbalance can pose challenges for machine learning algorithms, as they may tend to favor the majority class and overlook the minority class.

**Consequences of Data Imbalance**
-	Poor classification performance: Machine learning algorithms may overfit to the majority class and neglect the minority class, resulting in poor classification accuracy for the minority class.
-	Biased predictions: The model's predictions may reflect the imbalance in the data, leading to biased outcomes. This can be particularly problematic in real-world applications where fairness and accuracy are crucial.
-	Overlooking rare events: In cases where the minority class represents rare events, the model may struggle to identify these events due to their underrepresentation in the data.

**Techniques to Handle Data Imbalance**
- Oversampling: Oversampling involves replicating data points from the minority class to increase its representation in the dataset. This can be done using techniques like random oversampling, synthetic minority oversampling technique (SMOTE), or adaptive synthetic sampling (ADASYN).
-	Undersampling: Undersampling involves reducing the number of data points from the majority class to balance the class distribution. This can be done using techniques like random undersampling or NearMiss.
-	Cost-sensitive learning: Cost-sensitive learning algorithms assign different misclassification costs to different classes, allowing the model to focus more on accurately classifying the minority class.
-	Ensemble methods: Ensemble methods, such as random forests and Xgboost, can be more robust to data imbalance than individual models.
- Algorithm tuning: Tuning the hyperparameters of the machine learning algorithm can also help improve its performance on imbalanced data.


## 8. Model Selection Approach:

1. First, I used a Binary logistic regression model that, given certain geo-coordinates, predicted the likelihood of whether the crime was related to drugs. This will assist the administration of San Francisco in allocating resources to specific areas according to the forecast. 94% was my starting accuracy, which was "suspiciously" high. Therefore, I investigated for bias and discovered that my target class was unbalanced, meaning that there were significantly more crimes unrelated to drugs than crimes linked to them. I, therefore used SMOTE to oversample our target class. Following this, we were 77% accurate, which made sense even though the AUC increased from 0.68786 to 0.69875.


![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/9a20121c-aeaf-4e93-bddd-2d25c97160f9)


 
Target class distribution before oversampling(left figure) 
Target class distribution after oversampling(right figure)
Either undersampling or oversampling can be used to correct this unequal distribution. Although undersampling could be effective, a lot of valuable data is lost in the process, which is why oversampling is preferred. The logistic regression technique finds significance in these data.
Synthetic minority oversampling technique, or SMOTE, is the oversampling method. Two causes are present. To begin with, SMOTE is a widely used oversampling method. Second, oversampling a minority class in an unbalanced dataset may cause the model to learn excessive details from a few cases. This is typically the result of using a straightforward method like adding minority data at random. Conversely, SMOTE discovers the characteristics of the minority data points' vicinity. This improves the model's ability to generalize.


![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/c1b11c59-5f7c-43ae-bde2-dd74d25454f0)

 
(AUC before oversampling)

 ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/effae7e3-20f1-4a14-a463-01a1c86fbbdf)

 
(AUC after SMOTE)

## 9. Model Training and Testing

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/9a4b639d-41eb-4075-b68f-58e88677a54a)

XGBoost outperformed other models in predicting crime likelihood.
The models I evaluated predict the probability of each crime category occurring at a given location (geo-coordinate) during the day or night within a specific police district (PdDistrict). By aggregating these probabilities across the specified geo-coordinates, I can identify high-crime neighborhoods and allocate government resources effectively.

**As I Trained data Without Oversampling, I got Better Results:**
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/91a29b13-fbb6-45b4-804e-3d08bc71569b) ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/ee0b9727-50c5-4adb-b392-6779f7736782)
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/0cad1188-3eeb-45f2-802d-ffeaad52be43) ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/593780ab-c6c9-445c-b467-245185c725cc)

The classifier can correctly identify positive examples slightly more often than it incorrectly identifies negative examples. However, it does not do a very good job of distinguishing between positive and negative examples. It is important to note that the performance of a classifier on an ROC curve can be affected by the class imbalance in the data.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/ef118554-df42-40ce-8b2e-cef13e93f2f5)



## 10. Conclusion

To wrap up, my investigation into San Francisco's opioid epidemic has provided significant insights. Through the application of advanced analytics and machine learning, I pinpointed high-risk areas, proposed potential Safe Injection Site locations, and predicted the likelihood of crime. Acknowledging the constraints of Binary Logistic Regression, I pivoted towards more nuanced models such as Random Forest and XGBoost, ultimately finding XGBoost to be particularly effective. These findings underscore the importance of adopting sophisticated strategies in resource allocation to effectively tackle the intricate challenges presented by the opioid crisis.

## 11. References
[1]	https://www.kqed.org/news/11766169/san-francisco-fentanyl-deaths-up-almost

[2]	https://www.sfchronicle.com/bayarea/article/Bay-Briefing-Fentanyl-epidemic-worsens-in-San-14032040.php

[3]	https://www.sfchronicle.com/bayarea/article/California-bill-allowing-San-Francisco-safe-13589277.php

[4]	https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Histo

[5]	https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/data

[6]	https://www.quora.com/What-is-the-meaning-of-min-max-normalization

