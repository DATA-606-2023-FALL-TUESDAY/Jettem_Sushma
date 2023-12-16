##				REPORT

## 1. Project Title:  Addressing the Opioid Epidemic: A National Public Health Challenge 
#### Domain: Government / Health

- **Author's Name** - Sushma Jettem
- **Prepared for** - UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- **Semester** - Fall 2023
  
- <a href="https://github.com/sushmajettem05"><img align="left" src="https://img.shields.io/badge/-GitHub-181717?logo=github&style=flat" alt="icon | GitHub"/></a>
-
- 

## 2. Background: Tackling the Opioid Crisis in the United States and San Francisco

In the United States, drug overdoses have become the primary cause of unintentional deaths, with opioids driving a significant portion of these fatalities. The historical "War on Drugs" initiative, initiated in the 1970s, has evolved over time, contributing to an opioid crisis across several states. San Francisco, known for pioneering progressive health solutions, is currently facing a substantial drug issue and opioid crisis.

To combat this, the city has embraced innovative approaches, including Safe Injection Sites (SIS). SIS provides supervised spaces for individuals to consume intravenous drugs, aiming to reduce public disturbances associated with drug use. San Francisco's commitment to harm reduction aligns with its history of implementing forward-thinking public health measures. As we delve into this paper, we'll explore the origins of the opioid crisis, the impact of the "War on Drugs," and the novel strategies, such as Safe Injection Sites, implemented by San Francisco to address this pressing public health challenge.

## Research question: 
- Analyzing various crime categories across diverse neighborhoods: What are the top 5 neighborhoods with the highest rates of assaults? Are there specific combinations of crimes that commonly occur together in particular areas?
- Identifying potential locations for implementing Safe Injection Sites (SIS) on behalf of the San Francisco government.
- Do specific demographic groups or geographical areas experience a higher incidence of fatal opioid overdoses? 
- If such disparities exist, is there a discernible connection between prescription practices and opioid-related deaths within these demographics or regions?


## 3. Data:
- **Data sources/tools:** 1. https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry/data
2) https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/data

The San Francisco Police Department's database serves as the repository for collecting crime data. This historical dataset spans from January 2003 to May 2018. 
**Data Size:** - 219.7 MB
**Data shape** -
  - Rows = 2,215,024
  - Columns = 13 

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/34253b2a-7d46-447a-a533-b49611b60108)
 
## Data Info:
Below we can see that Variables and its types Associated with the data and data size.
 

Below we can see Different Category of Crimes and its Reports.
 

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/59cc821b-1d41-4206-9120-193a34d2e441)


### Let’s Focus on Drugs:
 

Types of Districts associated with Drugs:
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/82d19b1c-8052-4b6d-bb2d-5cb212d1deda)


- **Target Variable(s)** - Category, Drug, Crime.

# Literature Review
We utilized a random forest classifier to forecast opioid dependence according to ICD-9 codes, achieving an impressive F1 score of 0.776. Che et al. employed a recurrent neural network (RNN) to categorize opioid users as long-term users, short-term users, or opioid-dependent patients using diagnostic, procedural, and prescription data. Their top AUCROC score for detecting opioid-dependent patients (OD) reached approximately 0.8.
In comparison to prior research, our study has successfully addressed its limitations. Most previous approaches relied on clinical expertise for feature engineering, which could be cumbersome and incomplete. Our model harnesses the data-processing capabilities of deep learning, allowing us to incorporate a broader range of clinical features and uncover unexpected associations. Additionally, our use of the LSTM model enhances our capacity to retain information from lengthy sequences while mitigating the vanishing gradient issue.
Balancing precision and recall involve a trade-off. Adjusting model parameters can increase precision at the cost of reduced recall and vice versa. The F-1 score, as a weighted combination of these metrics, provides a more informative measure for evaluating predictive performance. Consequently, in this study, we prioritize the F-1 score as the primary metric for guiding model development and assessment.

## Analysis Approach and Inferences:
1. For every type of crime, we tallied the incidents and created a plot. The distribution was skewed, so we took the log to normalise it. The normalised distribution of crime categories is shown below. 

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/94865780-ed5d-4a2d-b49d-9656d4e49e97)


2. There were 915 different criminal descriptors; these descriptions help identify whether a crime involves drugs or not. In order to create the cluster maps, we tallied the occurrences for each crime description, filtered out those that fell below the 97th percentile, and retained the remaining ones.
3. To investigate the distributions of various categories (i.e., crimes throughout each PdDistrict, or Police District), a cluster map was created. Once more, the skewed distribution had an impact on our model, which is displayed below.Grand Theft Auto is an outlier, as can be seen, and we don't get any other information, therefore normalisation was necessary.
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/f9dc3e97-9671-4247-aa30-667ae0f6a3dc)


4. Therefore, min-max normalisation was used for normalisation since logarithmic normalisation loses the scale that indicates how big or tiny a feature is in relation to another. The normalised cluster map is shown below.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/d5c99199-0f66-4f55-86b8-d8d132a2a428)

 
## Here we can observe the following:
(a)Southern: extremely high occurrences of theft, including theft from auto
(b)Bayview: significant occurrences of violences and threats
(c)Tenderloin: appears to be an anomaly, with incredibly high rates of drug paraphernalia possession. Though it might be a false positive (i.e., these could be related to marijuana), tenderloin appears to be a good candidate for SIS installation. Therefore, more research is required.
5. Subsequently, we employed regular expressions and string pattern matching to filter crimes linked to drugs, and we tallied the instances of each unique description of drugs. The cluster map below was impacted by the distribution's skew once more. Tenderloin is an anomaly, as can be seen, and we learn nothing more..
  
 ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/858bbc2f-a486-420e-992c-0e974c049106)

 
6.Below is normalized cluster-map which shows distribution of narcotics related crimes across each PdDistrict.
 ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/365fe16e-bee8-4147-831a-3f186b51215c)

Inference:From the cluster-map above we can clearly conclude that, Tenderloin, Southern, Mission and Northern are the optimal candidates for installing SIS

7.  Time-series analysis was then carried out to examine opioid patterns over time. Initially, we created opioid categories and features by compressing a large number of criminal descriptions pertaining to drugs (such as barbiturate, coke, marijuana, meth, etc.). Next, we made a 30-day frame for every group, and from January 1, 2003, to May 15, 2018, we counted the number of occurrences for every group during this 30-day window, or every month. In order to eliminate monthly cyclical patterns, we indexed the months from 0 to 187. To illustrate these trends, a stacked histogram is shown below. As you can see, there was a notable increase in instances involving meth and heroin.
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/4fb42740-bf97-44ba-9474-4645950801d6)


8.To make the trends clearer, below is the normalized distribution of opioid trends across the years from 2003 to 2018.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/96bb820f-0d21-43e3-9149-f6d24a5c6006)

 
It is evident that over this time, the number of crack-related incidents decreased. In a similar vein, occurrences using marijuana decreased following its legalisation in 2016. However, crimes involving meth and heroin have increased dramatically; this is strong indication that the problem is widespread.




## Model Selection Approach:
1. First, we used a Binary logistic regression model that, given certain geo-coordinates, predicted the likelihood of whether or not the crime was related to drugs. This will assist the administration of San Francisco in allocating resources to specific areas according to the forecast. 94% was our starting accuracy, which was "suspiciously" high. Therefore, we investigated for bias and discovered that our target class was unbalanced, meaning that there were significantly more crimes unrelated to drugs than crimes linked to them. We therefore used SMOTE to oversample our target class. Following this, we were 77% accurate, which made sense even though the AUC increased from 0.68786 to 0.69875.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/f9cb2124-4c8f-4a39-89e8-bd122d157fd3)

 
Target class distribution before oversampling(left figure) 
Target class distribution after oversampling(right figure)
Either undersampling or oversampling can be used to correct this unequal distribution. Although undersampling could be effective, a lot of valuable data is lost in the process, which is why oversampling is preferred. The logistic regression technique finds significance in these data.
Synthetic minority oversampling technique, or SMOTE, is the oversampling method. Two causes are present. To begin with, SMOTE is a widely used oversampling method. Second, oversampling a minority class in an unbalanced dataset may cause the model to learn excessive details from the few cases. This is typically the result of using a straightforward method like adding minority data at random. Conversely, SMOTE discovers the characteristics of the minority data points' vicinity. This improves the model's ability to generalise.

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/c1b11c59-5f7c-43ae-bde2-dd74d25454f0)

 
(AUC before oversampling)

 ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/effae7e3-20f1-4a14-a463-01a1c86fbbdf)

 
(AUC after SMOTE)

#Inference: 
We came to the conclusion that our issue required more than a binary logistic regression, or Binary LR. For example, we might have a very high false positive rate if we allocated government resources based on these projections. We needed to go further since simply identifying geo-coordinates or regions with high rates of drug-related or non-narcotics-related crime is insufficient. For example, we would prefer to devote more resources in areas with high rates of murder than arson. Consequently, it is possible to train other multi-class models as Random Forest, XGBoost, KNN, and Multinomial Logistic Regression (Multinomial LR).
