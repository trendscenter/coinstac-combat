# coinstac-combat
Coinstac Code for ComBat Algorithm with simulated data

Tools: Python 3.6.5, coinstac-simulator 4.2.0

Steps
1) sudo npm i -g coinstac-simulator@4.2.0
2) git clone https://github.com/trendscenter/coinstac-combat
3) cd coinstac-combat
4) docker build -t coinstac/combat .
5) coinstac-simulator


How to run ComBat-DC Harmonization in COINSTAC
How to give input?
All input files should be in CSV format. Other formats may produce unexpected results.
There are two types of input file.
1) Data : This file contains the feature information. Each row represents a
2) Covariates : This file contains the demographic information e.g., gender, diagnosis, age etc. This information is provided so that Harmonization preserves them throughout the process.
Do we need to add a label in the input file?
No
How to give the site information?
No need to provide any site information explicitly when you are using COINSTAC. Each node should contain data from a single site.
What is the Output?
After harmonization, COINSTAC will produce harmonized data which will have similar dimension as input data.
Does harmonization remove the biological covariates?
No, ComBat does not remove any biological covariates such as gender, age and other information. Any biological information provided in the covariates file will be preserved during the harmonization process.
Example Case:
If we have data collected from 3 sites. Then there should be three nodes in the COINSTAC. Each node will upload two input files containing data and demographic information. Then running the ComBat Harmonization will automatically harmonize each site accordingly.
Steps for running the algorithm:
1) Separate data accordingly to the site information. Each input file should contain only a single site information.
2) Setup N number of nodes in the COINSTAC. ‘N’ is the number of sites where data was collected.
3) Upload files for each node.
4) Run the analysis.
5) Collect the harmonized data and conduct further analysis.


# Acknowledgement:

Jean-Philippe Fortin for the sharing the cetralized version of the code. 


