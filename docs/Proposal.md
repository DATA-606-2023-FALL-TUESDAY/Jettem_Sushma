# Proposal 

Addressing the Opioid Epidemic: A National Public Health Challenge

Domain: Government / Health
Abstract:
In the United States, drug overdoses have emerged as the primary cause of unintentional fatalities, accounting for 52,404 lethal incidents in 2015. Opioid addiction serves as the driving force behind this epidemic, contributing to 20,101 overdose deaths linked to prescription pain relievers and 12,990 overdose deaths associated with heroin in the same year (CDC, ASAM). The effective distribution and administration of Narcan (naloxone HCl) plays a pivotal role in reducing mortality rates resulting from opioid overdoses.
The term "War on Drugs" describes a government-led effort initiated in the 1970s to combat illegal drug usage, distribution, and trade by intensifying penalties for those involved. This initiative has continued to evolve over the years and has resulted in an opioid crisis in several US states. Presently, there is a contentious debate about whether this opioid crisis is primarily a result of Mexican and Central American migration or stems from the deregulation of pharmaceutical companies and shortcomings in the private healthcare system. At this moment, San Francisco is confronting a significant drug issue and opioid crisis.
Research question: 
•	Analyzing various crime categories across diverse neighborhoods: What are the top 5 neighborhoods with the highest rates of assaults? Are there specific combinations of crimes that commonly occur together in particular areas?
•	Identifying potential locations for implementing Safe Injection Sites (SIS) on behalf of the San Francisco government.
•	Do specific demographic groups or geographical areas experience a higher incidence of fatal opioid overdoses? 
•	If such disparities exist, is there a discernible connection between prescription practices and opioid-related deaths within these demographics or regions?

Introduction:
	Natural derivatives of Opium like heroin are called Opiates which are illegal. Similar synthetically synthesized drugs have been put under the class of Opioids which are legally available. Opioids are prescribed primarily as pain relievers despite a high risk of addiction and overdose. The increase in deaths caused by the risks involved with the consumption of opioids was alarming and declared an epidemic.
San Francisco (SF) has a rich history of pioneering progressive public health solutions, such as medical cannabis and needle exchange programs, even before they were legally accepted or widely adopted. This commitment to innovation is exemplified by California passing a bill that permits SF to establish 
Safe Injection Sites (SIS).
Safe Injection Sites (SIS): Safe Injection Sites are supervised medical facilities designed to offer a clean and supportive environment where individuals can safely consume intravenous recreational drugs, thus reducing public drug use-related disturbances. These sites are part of a harm reduction strategy in addressing drug-related issues. The first SIS in North America was established in the Downtown Eastside (DTES) neighborhood of Vancouver in 2003.

Dataset:
Data sources/tools: 
•	https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry/data
•	https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/data

The database of the San Francisco police department is used to collect data. Data on crimes from January 2003 to May 2018 is historical. There are 2215024 rows and 13 columns in the dataset.

 ![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/b709e51f-e2e5-4a25-ad5d-1e23e4d4f5eb)

Data Info:
Below we can see that Variables and its types Associated with the data and data size.
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/9f0a2f25-2e40-46cb-937e-026ded18f0e0)

Below we can see Different Category of Crimes and its Reports.
 

![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/0c43ac06-c737-4768-9665-bd8191c5dc9e)





Let’s Focus on Drugs:
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/d38ff480-bdfc-4be2-8d84-60dbdf9fe71f)

Types of Districts associated with Drugs:
 
![image](https://github.com/DATA-606-2023-FALL-TUESDAY/Jettem_Sushma/assets/144371682/5b5d3fcc-7223-4c9f-8504-6bd5e6c0cfdc)

Target Variables:
•	Examining the relationship between crime types and neighborhoods spanning from 2003 to 2018. Are there specific crime categories that tend to happen together frequently in particular areas?
•	Assessing the correlation between drug usage patterns and neighborhoods from 2013 to 2018. Identifying prospective neighborhoods or regions where the San Francisco government could establish safe injection sites.
•	Forecasting the specific type or category of crime based on the spatial and temporal characteristics provided.

We’ll be training multiple ML models to train and explore the results and analysis.


