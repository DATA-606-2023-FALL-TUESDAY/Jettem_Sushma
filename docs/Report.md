#				REPORT

## Addressing the Opioid Epidemic: A National Public Health Challenge

## Domain: Government / Health
## Abstract:
In the United States, drug overdoses have emerged as the primary cause of unintentional fatalities, accounting for 52,404 lethal incidents in 2015. Opioid addiction serves as the driving force behind this epidemic, contributing to 20,101 overdose deaths linked to prescription pain relievers and 12,990 overdose deaths associated with heroin in the same year (CDC, ASAM). The effective distribution and administration of Narcan (naloxone HCl) plays a pivotal role in reducing mortality rates resulting from opioid overdoses.
The term "War on Drugs" describes a government-led effort initiated in the 1970s to combat illegal drug usage, distribution, and trade by intensifying penalties for those involved. This initiative has continued to evolve over the years and has resulted in an opioid crisis in several US states. Presently, there is a contentious debate about whether this opioid crisis is primarily a result of Mexican and Central American migration or stems from the deregulation of pharmaceutical companies and shortcomings in the private healthcare system. At this moment, San Francisco is confronting a significant drug issue and opioid crisis.
## Research question: 
•	Analyzing various crime categories across diverse neighborhoods: What are the top 5 neighborhoods with the highest rates of assaults? Are there specific combinations of crimes that commonly occur together in particular areas?
•	Identifying potential locations for implementing Safe Injection Sites (SIS) on behalf of the San Francisco government.
•	Do specific demographic groups or geographical areas experience a higher incidence of fatal opioid overdoses? 
•	If such disparities exist, is there a discernible connection between prescription practices and opioid-related deaths within these demographics or regions?

## Introduction:
	Natural derivatives of Opium like heroin are called Opiates which are illegal. Similar synthetically synthesized drugs have been put under the class of Opioids which are legally available. Opioids are prescribed primarily as pain relievers despite a high risk of addiction and overdose. The increase in deaths caused by the risks involved with the consumption of opioids was alarming and declared an epidemic.
San Francisco (SF) has a rich history of pioneering progressive public health solutions, such as medical cannabis and needle exchange programs, even before they were legally accepted or widely adopted. This commitment to innovation is exemplified by California passing a bill that permits SF to establish. 
Safe Injection Sites (SIS).
Safe Injection Sites (SIS): Safe Injection Sites are supervised medical facilities designed to offer a clean and supportive environment where individuals can safely consume intravenous recreational drugs, thus reducing public drug use-related disturbances. These sites are part of a harm reduction strategy in addressing drug-related issues. The first SIS in North America was established in the Downtown Eastside (DTES) neighborhood of Vancouver in 2003.

# Dataset:
## Data sources/tools: 
•	https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry/data
•	https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/data

The database of the San Francisco police department is used to collect data. Data on crimes from January 2003 to May 2018 is historical. There are 2215024 rows and 13 columns in the dataset.

 
## Data Info:
Below we can see that Variables and its types Associated with the data and data size.
 

Below we can see Different Category of Crimes and its Reports.
 






Let’s Focus on Drugs:
 

Types of Districts associated with Drugs:
 

## Target Variables:
•	Examining the relationship between crime types and neighborhoods spanning from 2003 to 2018. Are there specific crime categories that tend to happen together frequently in particular areas?
•	Assessing the correlation between drug usage patterns and neighborhoods from 2013 to 2018. Identifying prospective neighborhoods or regions where the San Francisco government could establish safe injection sites.
•	Forecasting the specific type or category of crime based on the spatial and temporal characteristics provided.






Literature Review
We utilized a random forest classifier to forecast opioid dependence according to ICD-9 codes, achieving an impressive F1 score of 0.776. Che et al. employed a recurrent neural network (RNN) to categorize opioid users as long-term users, short-term users, or opioid-dependent patients using diagnostic, procedural, and prescription data. Their top AUCROC score for detecting opioid-dependent patients (OD) reached approximately 0.8.
In comparison to prior research, our study has successfully addressed its limitations. Most previous approaches relied on clinical expertise for feature engineering, which could be cumbersome and incomplete. Our model harnesses the data-processing capabilities of deep learning, allowing us to incorporate a broader range of clinical features and uncover unexpected associations. Additionally, our use of the LSTM model enhances our capacity to retain information from lengthy sequences while mitigating the vanishing gradient issue.
Balancing precision and recall involve a trade-off. Adjusting model parameters can increase precision at the cost of reduced recall and vice versa. The F-1 score, as a weighted combination of these metrics, provides a more informative measure for evaluating predictive performance. Consequently, in this study, we prioritize the F-1 score as the primary metric for guiding model development and assessment.

## Analysis Approach and Inferences:
1. For every type of crime, we tallied the incidents and created a plot. The distribution was skewed, so we took the log to normalise it. The normalised distribution of crime categories is shown below. 
2. There were 915 different criminal descriptors; these descriptions help identify whether a crime involves drugs or not. In order to create the cluster maps, we tallied the occurrences for each crime description, filtered out those that fell below the 97th percentile, and retained the remaining ones.
3. To investigate the distributions of various categories (i.e., crimes throughout each PdDistrict, or Police District), a cluster map was created. Once more, the skewed distribution had an impact on our model, which is displayed below.Grand Theft Auto is an outlier, as can be seen, and we don't get any other information, therefore normalisation was necessary.
 

4. Therefore, min-max normalisation was used for normalisation since logarithmic normalisation loses the scale that indicates how big or tiny a feature is in relation to another. The normalised cluster map is shown below.
 
## Here we can observe the following:
(a)Southern: extremely high occurrences of theft, including theft from auto
(b)Bayview: significant occurrences of violences and threats
(c)Tenderloin: appears to be an anomaly, with incredibly high rates of drug paraphernalia possession. Though it might be a false positive (i.e., these could be related to marijuana), tenderloin appears to be a good candidate for SIS installation. Therefore, more research is required.
5. Subsequently, we employed regular expressions and string pattern matching to filter crimes linked to drugs, and we tallied the instances of each unique description of drugs. The cluster map below was impacted by the distribution's skew once more. Tenderloin is an anomaly, as can be seen, and we learn nothing more..
  
 
 
6.Below is normalized cluster-map which shows distribution of narcotics related crimes across each PdDistrict.
 
Inference:From the cluster-map above we can clearly conclude that, Tenderloin, Southern, Mission and Northern are the optimal candidates for installing SIS

7.  Time-series analysis was then carried out to examine opioid patterns over time. Initially, we created opioid categories and features by compressing a large number of criminal descriptions pertaining to drugs (such as barbiturate, coke, marijuana, meth, etc.). Next, we made a 30-day frame for every group, and from January 1, 2003, to May 15, 2018, we counted the number of occurrences for every group during this 30-day window, or every month. In order to eliminate monthly cyclical patterns, we indexed the months from 0 to 187. To illustrate these trends, a stacked histogram is shown below. As you can see, there was a notable increase in instances involving meth and heroin.
 

8.To make the trends clearer, below is the normalized distribution of opioid trends across the years from 2003 to 2018.
 
It is evident that over this time, the number of crack-related incidents decreased. In a similar vein, occurrences using marijuana decreased following its legalisation in 2016. However, crimes involving meth and heroin have increased dramatically; this is strong indication that the problem is widespread.




## Model Selection Approach:
1. First, we used a Binary logistic regression model that, given certain geo-coordinates, predicted the likelihood of whether or not the crime was related to drugs. This will assist the administration of San Francisco in allocating resources to specific areas according to the forecast. 94% was our starting accuracy, which was "suspiciously" high. Therefore, we investigated for bias and discovered that our target class was unbalanced, meaning that there were significantly more crimes unrelated to drugs than crimes linked to them. We therefore used SMOTE to oversample our target class. Following this, we were 77% accurate, which made sense even though the AUC increased from 0.68786 to 0.69875.
 
Target class distribution before oversampling(left figure) 
Target class distribution after oversampling(right figure)
Either undersampling or oversampling can be used to correct this unequal distribution. Although undersampling could be effective, a lot of valuable data is lost in the process, which is why oversampling is preferred. The logistic regression technique finds significance in these data.
Synthetic minority oversampling technique, or SMOTE, is the oversampling method. Two causes are present. To begin with, SMOTE is a widely used oversampling method. Second, oversampling a minority class in an unbalanced dataset may cause the model to learn excessive details from the few cases. This is typically the result of using a straightforward method like adding minority data at random. Conversely, SMOTE discovers the characteristics of the minority data points' vicinity. This improves the model's ability to generalise.
 
(AUC before oversampling)

 
 
(AUC after SMOTE)

Inference: We came to the conclusion that our issue required more than a binary logistic regression, or Binary LR. For example, we might have a very high false positive rate if we allocated government resources based on these projections. We needed to go further since simply identifying geo-coordinates or regions with high rates of drug-related or non-narcotics-related crime is insufficient. For example, we would prefer to devote more resources in areas with high rates of murder than arson. Consequently, it is possible to train other multi-class models as Random Forest, XGBoost, KNN, and Multinomial Logistic Regression (Multinomial LR).
