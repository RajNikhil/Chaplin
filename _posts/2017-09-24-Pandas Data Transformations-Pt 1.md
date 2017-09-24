
# Data Manipulation with Pandas
***

### Introduction

Pandas is by far the most matured and feature rich python library for data manipulation. It is used by numerous data analysts and data scientists in preparing data for any kind of analysis.

Data preparation is a pretty intensive job. It is no wonder then about 75 to 80 percent of time it takes to build a model is dedicated to data preparation. However, data preparation can involve from a few to several steps depending on what the structure of the raw data is and how much it needs to be transformed in order to make it useful for analysis. Most of the steps involved may be unique to a given analysis. 

In this first of several part post, I will cover some of the data transformation I have done recently in some of my data analyses projects. These are only a few of the many data manipulations I have done and I will post others with a different dataset involving columns of different datatypes.
***

As always, begin with importing the necessary libraries. Since, the only steps that will be carried out here are data transformation just Pandas and Numpy packages are imported. The dataset used in here is for [US Technology Jobs on Dice.com](https://www.kaggle.com/PromptCloudHQ/us-technology-jobs-on-dicecom) and is available for download at Kaggle.


```python
import pandas as pd
import numpy as np
```


```python
df=pd.read_csv('dice_com-job_us_sample.csv')
```

Once the data is imported and assigned to a variable, 'df' in this case, the following four steps are carried out at the beginning:
1. View part of the data with ```DataFrame.head()``` function
2. Determine the depth (or number of rows) of the dataset with the use of length function
3. Confirm datatypes of the columns in the dataset with ```DataFrame.dtypes```
4. Finally, count null values (if any) in all the columns of the dataset by combining isnull and sum functions


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>advertiserurl</th>
      <th>company</th>
      <th>employmenttype_jobstatus</th>
      <th>jobdescription</th>
      <th>jobid</th>
      <th>joblocation_address</th>
      <th>jobtitle</th>
      <th>postdate</th>
      <th>shift</th>
      <th>site_name</th>
      <th>skills</th>
      <th>uniq_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.dice.com/jobs/detail/AUTOMATION-TE...</td>
      <td>Digital Intelligence Systems, LLC</td>
      <td>C2H Corp-To-Corp, C2H Independent, C2H W2, 3 M...</td>
      <td>Looking for Selenium engineers...must have sol...</td>
      <td>Dice Id : 10110693</td>
      <td>Atlanta, GA</td>
      <td>AUTOMATION TEST ENGINEER</td>
      <td>1 hour ago</td>
      <td>Telecommuting not available|Travel not required</td>
      <td>NaN</td>
      <td>SEE BELOW</td>
      <td>418ff92580b270ef4e7c14f0ddfc36b4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.dice.com/jobs/detail/Information-S...</td>
      <td>University of Chicago/IT Services</td>
      <td>Full Time</td>
      <td>The University of Chicago has a rapidly growin...</td>
      <td>Dice Id : 10114469</td>
      <td>Chicago, IL</td>
      <td>Information Security Engineer</td>
      <td>1 week ago</td>
      <td>Telecommuting not available|Travel not required</td>
      <td>NaN</td>
      <td>linux/unix, network monitoring, incident respo...</td>
      <td>8aec88cba08d53da65ab99cf20f6f9d9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.dice.com/jobs/detail/Business-Solu...</td>
      <td>Galaxy Systems, Inc.</td>
      <td>Full Time</td>
      <td>GalaxE.SolutionsEvery day, our solutions affec...</td>
      <td>Dice Id : CXGALXYS</td>
      <td>Schaumburg, IL</td>
      <td>Business Solutions Architect</td>
      <td>2 weeks ago</td>
      <td>Telecommuting not available|Travel not required</td>
      <td>NaN</td>
      <td>Enterprise Solutions Architecture, business in...</td>
      <td>46baa1f69ac07779274bcd90b85d9a72</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.dice.com/jobs/detail/Java-Develope...</td>
      <td>TransTech LLC</td>
      <td>Full Time</td>
      <td>Java DeveloperFull-time/direct-hireBolingbrook...</td>
      <td>Dice Id : 10113627</td>
      <td>Bolingbrook, IL</td>
      <td>Java Developer (mid level)- FT- GREAT culture,...</td>
      <td>2 weeks ago</td>
      <td>Telecommuting not available|Travel not required</td>
      <td>NaN</td>
      <td>Please see job description</td>
      <td>3941b2f206ae0f900c4fba4ac0b18719</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.dice.com/jobs/detail/DevOps-Engine...</td>
      <td>Matrix Resources</td>
      <td>Full Time</td>
      <td>Midtown based high tech firm has an immediate ...</td>
      <td>Dice Id : matrixga</td>
      <td>Atlanta, GA</td>
      <td>DevOps Engineer</td>
      <td>48 minutes ago</td>
      <td>Telecommuting not available|Travel not required</td>
      <td>NaN</td>
      <td>Configuration Management, Developer, Linux, Ma...</td>
      <td>45efa1f6bc65acc32bbbb953a1ed13b7</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Rows in the data file
len(df.index)
```




    22000




```python
#datatypes of all the columns in a dataset
df.dtypes
```




    advertiserurl               object
    company                     object
    employmenttype_jobstatus    object
    jobdescription              object
    jobid                       object
    joblocation_address         object
    jobtitle                    object
    postdate                    object
    shift                       object
    site_name                   object
    skills                      object
    uniq_id                     object
    dtype: object




```python
#Check for Null values in all the columns
df.isnull().sum()
```




    advertiserurl                   0
    company                        50
    employmenttype_jobstatus      230
    jobdescription                  0
    jobid                           0
    joblocation_address             3
    jobtitle                        0
    postdate                        0
    shift                         357
    site_name                   18510
    skills                         43
    uniq_id                         0
    dtype: int64



### *Fill Null Values*

Once the columns with null values they can be filled using the ```DataFrame.fillna()``` function. It is possible to fill null values in all the columns in a single step or do it invidually one column at a time. Below is a demonstration of both these methods.


```python
#Fill null values in individual columns
df['company']=df['company'].fillna(value='Unknown')
```


```python
df.isnull().sum()
```




    advertiserurl                   0
    company                         0
    employmenttype_jobstatus      230
    jobdescription                  0
    jobid                           0
    joblocation_address             3
    jobtitle                        0
    postdate                        0
    shift                         357
    site_name                   18510
    skills                         43
    uniq_id                         0
    dtype: int64




```python
df=df.fillna(value='Unknown')
```


```python
df.isnull().sum()
```




    advertiserurl               0
    company                     0
    employmenttype_jobstatus    0
    jobdescription              0
    jobid                       0
    joblocation_address         0
    jobtitle                    0
    postdate                    0
    shift                       0
    site_name                   0
    skills                      0
    uniq_id                     0
    dtype: int64



### *List Unique Values*

Quite often it proves useful to look at the unique values/categories in a column. This can be used to verify if the data extracted was right or if there are any unexpected values. It is also helpful in situations where one wants to confirm if the filtered dataset was in fact filtered correctly. Of course there may be other applications of this depending on the requirement.

Below I've used the ```length()``` and ```unique()``` functions to understand the column company i.e. know how many unique companies are in the data set, using ```length()``` and ```unique()``` function together and know what are the names of those companies, using just the ```unique()``` function.


```python
#Count and List of Unique values in each column
len(df['company'].unique())
```




    4292




```python
#Names of the companies in the dataset
list(df['company'].unique())
```




    ['Digital Intelligence Systems, LLC',
     'University of Chicago/IT Services',
     'Galaxy Systems, Inc.',
     'TransTech LLC',
     'Matrix Resources',
     'Yash Technologies',
     'Noble1',
     'Bluebeam Software, Inc.',
     'Genesis10',
     'VanderHouwen & Associates, Inc.',
     'Maxonic, Inc.',
     'CSI (Consultant Specialists Inc.)',
     'Eastridge Workforce Solutions',
     'Collabera',
     'Fahrenheit IT Staffing & Consulting',
     'Avesta Computer Services',
     'Amazon',
     'ReqRoute, Inc',
     'Turnberry Solutions',
     'Alpha Recruitment',
     'Etouch Systems Corp',
     'Centizen',
     'Mygo Consulting',
     'Alpha Consulting Corp.',
     'Ascent',
     'Amerit Consulting',
     'Tellus Solutions',
     'Stratitude Inc',
     'Xoriant Corporation',
     'iSpace, Inc',
     'Akraya, Inc.',
     'Thinkfind Corporation',
     'Precision Task Group',
     'Tailwind Associates',
     'Princeton Information Ltd',
     'SMCI',
     'TAD PGS, Inc',
     'Mitchell Martin',
     'MSRCosmos',
     'TM Floyd',
     'P. Murphy & Associates, Inc',
     'TRIGYN TECHNOLOGIES, INC.',
     'SIS-Systems Integration Solutions, Inc.',
     'Denali Advanced Integration',
     'DBA Web Technologies',
     'eXcell',
     'Chenoa Information Services',
     'Irvine Technology Corporation',
     'Staff Tech',
     'BayOne Solutions',
     'Amadan Recruitment',
     'Sumeru',
     'Staff Matters',
     'Fourth Technologies, Inc.',
     'Redstream Technology',
     'The Armada Group',
     'HEAT Software',
     'Metropolitan Washington Airports Authority',
     'Mordue, Allen, Roberts, Bonney, Ltd',
     'Tech Observer',
     'Cystems Logic',
     'I.T. Solutions, Inc.',
     'Travelclick',
     'Verys',
     'Software Guidance & Assistance',
     'John Muir Health',
     'Gunther Douglas, Inc.',
     'PWC',
     'UnitedHealth Group',
     'First Tek, Inc.',
     'First Republic Bank',
     'Epsilon',
     'IDC Technologies',
     'Aegistech Inc.',
     'Target',
     'Wimmer Solutions',
     'Genuent, LLC',
     'Citrix',
     'Accenture',
     'Albano Systems Inc',
     'CyberCoders',
     'ICST, LLC',
     'Independent Software',
     'New Age Software Services, Inc',
     'Deloitte',
     'iTech Solutions, Inc',
     'CVS Health',
     'PeopleServe, Inc.',
     'Cognizant Technology Solutions',
     '3i Infotech Inc.',
     'Incendia Partners',
     'ASRC Federal',
     'Hewlett Packard Enterprise Company',
     'Spruce Technology Inc.',
     'Aquinas Consulting',
     'Bivium Group',
     'Redolent, Inc',
     'Network Objects Inc.',
     'Boston Technology Corporation',
     'Corporate Systems Associates',
     'eclaro',
     'Request Technology, LLC',
     'IT First Source',
     'Primus Software Corp',
     'Sunrise Systems, Inc.',
     'ADPI, LLC..',
     'Informatica',
     'E*Pro, Inc.',
     'Entelli Consulting LLC',
     'Addison Group',
     'Tekmark Global Solutions LLC',
     'The Robinson Group',
     'Volt',
     'COOLSOFT',
     'Multivision, Inc.',
     'Queen Consulting Group',
     'HCL America Inc.',
     'CODEFORCE 360',
     'Exafluence',
     'Maxima Consulting Inc',
     'LJP Services, Inc',
     'Scadea Solutions Inc',
     'Agile Enterprise Solutions, Inc.',
     'Henry Elliott & Company Inc.',
     'Strategic Employment Partners',
     'Next Step Systems',
     'Avance Consulting',
     'N-Tier Solutions Inc.',
     'Bridge Technical Talent',
     'GlobalLogic, Inc.',
     'V2Soft',
     'Chase Technology Consultants',
     'Kaizen Technologies',
     'Trident Consulting Inc.',
     'InfoVision, Inc.',
     'Shivam Infotech',
     '9to9 Software Solutions LLC',
     'Collab Infotech',
     'Technosoft Corporation',
     'Aureus Group',
     'Will-N-Able',
     'SoftHQ, Inc.',
     'Casa Systems',
     'The Creative Group',
     'Corporate Consulting Services',
     'Net2Source Inc.',
     'OmniPoint Staffing',
     'SolutionIT, Inc.',
     'Rang Technologies Inc.',
     'Cannon Search Partners',
     'Infosoft Inc',
     'The Tri-Com Consulting Group',
     'Cyma Systems Inc',
     'Centage Corporation',
     'PeopleCom, Inc.',
     'Apex Systems, Inc',
     'Plymouth Rock Assurance Corporation',
     'ManageForce Corporation',
     'TECHstaff Resources, Inc.',
     'VED Software Services, Inc.',
     'LevelUp Force, LLC',
     'Core Infotechnology',
     'GlobeSoft Resources',
     'Newt Global',
     'Puresoft, Inc.',
     'Bravens Inc.',
     'Cypress Group',
     'ICONMA',
     'Endure Technology Solutions, Inc.',
     'CTP - Cloud Technology Partners',
     'Data Net IT',
     'Encore Consulting Services',
     'VINFORMAX',
     'Tasacom Technologies',
     'Booz Allen Hamilton',
     'IT Trailblazers, LLC',
     'CenturyLink',
     'Global Technical Talent',
     'American IT Resource Group Inc.',
     'Connected Systems',
     'Zoom Technical Services, Inc.',
     's.com',
     'Data Based Development Systems',
     'IT People Corporation',
     'U.S. Tech Solutions Inc.',
     'Cengage Learning',
     'Comrise',
     'Execu/Search Group',
     'Invaluable',
     'The CEI Group',
     'Idhasoft',
     'Lightstat',
     'NTT DATA, Inc.',
     'Sirius Computer Solutions Inc',
     'Softpath System, LLC.',
     'Paladin Consulting, Inc.',
     'Ness Software Engineering Services',
     'Saint Anselm College',
     'Advancement Alternatives',
     'PFI Tech',
     'Red Zone Resources',
     'The FootBridge Companies',
     'Celestos, LLC',
     'Enterprise Solutions',
     'Bay State Search',
     'City Of Cambridge',
     'SRISYS Inc.',
     'Quintessence Computer Corp',
     'Five Star Senior Living',
     'Xceltrait Inc.',
     'IT Mantra',
     'Computer Merchant, Ltd., The',
     'Shain Associates',
     'Apps Associates',
     'Pro Search, Inc.',
     'ProTom International',
     'Softek International Inc.',
     'Xpeerant Inc.',
     'APN Consulting Inc',
     'Contemporaries',
     'Vertex Solutions Inc.',
     'RitePros Inc.',
     'Gardner Resources Consulting, LLC',
     'INSYS Group',
     'Now Business Intelligence',
     'Aditum',
     'The Vesume Group',
     'Cheshire Medical Center',
     'CSC',
     'Robert Half Technology',
     'Aries Systems Corporation',
     'BigR.io',
     'Softnice Inc',
     'HonorVet Technologies',
     'Alexander Technology Group',
     'John Galt Staffing',
     'Lawrence Harvey',
     'Madison Resources',
     'HCL Global Systems',
     'Edge Technology Services, Inc.',
     'M9 Solutions',
     'TSR Consulting Services, Inc.',
     'GRT Corporation',
     'The University of Connecticut',
     'Overture Partners, LLC',
     'S2Tech',
     'I and I Software Inc',
     'SeaGlass IT',
     'Market Street Talent',
     'Sricom, Inc.',
     'Stefanini, Inc.',
     'Reed Business Information',
     'CATIC',
     'Longford & Company',
     'SS & C Technologies Inc',
     'A2C Consulting',
     'Neoris',
     'InfoPeople Corp',
     'Encore Semi Inc',
     'FlaggStaff Technology Group, Inc',
     'MAPFRE U.S.A. Corp.',
     'PredicaInc',
     'Knowledge Momentum',
     'Lightwave Partners',
     'Braves Technologies LLC',
     'Bridge Search Associates',
     'Publishers Clearing House',
     'Compunnel Software Group Inc.',
     'Buxton Consulting',
     'Springfield Technical Community College',
     'Timon Group',
     'Beth Israel Deaconess Medical Center',
     'Nuance Communications, Inc.',
     'MassDOT',
     'The Kineta Group',
     'Business Knowledge Services',
     'Themesoft Inc',
     'CareCore | MedSolutions',
     'Black Diamond Networks',
     'Remark Staffing Specialists',
     'Dell',
     'ClientSolv Inc',
     'K Anand Corporation',
     'Scalable Systems',
     'Connection',
     'Virtusa',
     'MEMIC',
     'Centech Group, Inc.',
     'K2 Partnering Solutions, Inc.',
     'Resourcesys',
     'Kelly IT',
     'Confidential Company',
     'Simplion Technologies Inc',
     'DecisionWave',
     'Michael Anthony Associates Inc',
     'Wipro Ltd.',
     "Boston Children's Hospital",
     'TekShapers',
     'Ariston Tek, Inc.',
     'Partners HealthCare',
     'Staffmark',
     'Softsages LLC',
     'GCR Inc.',
     'OperationIT',
     'Stowe Group',
     'Proficient Business Systems Inc',
     'athenahealth, Inc.',
     'Smart Staffing Services DBA Smart-Tek',
     'Matrix Technology Group',
     'Northern Bank',
     'Tektree Systems Inc.',
     'Cardinal Technology Solutions',
     'NEOS LLC',
     'Connexions Data Inc',
     'Empower Professionals',
     'Thorndale Partners LLC',
     'The McIntyre Group',
     'SoftPath Technologies LLC',
     'Techgene Solutions LLC',
     'Equinoxys',
     'j2 Global',
     'Velocity Technology Resources',
     'FusionForte',
     'VoiceGlance',
     'Cyber 360 Solutions',
     'Cox Automotive',
     'AQUA Information Systems, Inc.',
     'oneZero Financial Systems',
     'IDEXX Laboratories',
     'The Wherry Group',
     'Ztek Consulting',
     'Information Technology Engineering Corporation',
     'JLG Technologies',
     'USM Business Systems',
     'Kelly Services',
     'ComTec Information Systems',
     'M & R Consultants Corporation',
     'eRichards Consulting',
     'CORE Higher Education Group',
     'Syrinx Consulting Corporation',
     'Waltech, Inc.',
     'Kaman Fuzing and Precision Products',
     'Dotcom Team, LLC',
     'Century Bank and Trust',
     'Tufts University',
     'Delphi-US',
     'ITECH Analyst Corp',
     'Cornerstone Technology Solutions',
     'Transperfect Translations',
     'FORTIRA INC.',
     'Analytic Recruiting Inc',
     'Central Business Solutions',
     'AGS International',
     'Adroit Resources',
     'IT Solutions',
     'Primesoft, Inc',
     'Gallop Solutions, Inc.',
     'Altimetrik',
     'AMS Staffing Inc.',
     'Stott and May',
     'W3Global',
     'Sysmind, LLC',
     'BuzzClan LLC',
     'Broadridge Financial Solutions, Inc',
     'mk North America, Inc.',
     'Marchon Partners',
     'WinterWyman',
     'PanAsia Resources Pte Ltd.',
     'SNI Companies dba SNI Technology',
     'Joss Data, Inc',
     'Spyglass Partners LLC',
     'AVID Technical Resources',
     'Kriya Software Solutions Inc',
     'Carbon Black, Inc.',
     'LoopPay',
     'CloudLabs, Inc.',
     'TDS: Transitional Data Services',
     'Vestmark, Inc.',
     'CoStar Realty Information, Inc',
     'SmartIMS Inc.',
     'Pegasystems',
     'FrameWork Development Group',
     'Unknown',
     'Oracle Corporation',
     'SPECTRAFORCE TECHNOLOGIES Inc.',
     'Raytheon',
     'The Matlen Silver Group, Inc.',
     'DocuTell Inc',
     'Dextro Software Systems Inc.',
     'Synergistic Systems, Inc.',
     'Advans IT Services',
     'Ascensus',
     'Carousel Industries',
     'BURGEON IT SERVICES LLC',
     'Phaidon International',
     'VLink Inc',
     'Talus Partners',
     'Hackett Group',
     'SolutionsIQ',
     'Source Infotech',
     'CGI',
     'Interpro Inc.',
     'Accelerated Innovators',
     'InfoTech Spectrum Inc',
     'SunTechPros, Inc.',
     'Company Confidential',
     'Computer Task Group, Inc',
     'IAR Systems Software, Inc',
     'Bernard, Nickels & Associates',
     'Futures Group IT LLC.',
     'Beacon Systems, Inc',
     'Incapsulate, LLC',
     'Useready',
     'Stream IT',
     'Clinton Savings Bank',
     'Computer Aid, Inc.',
     'Sacc, Inc.',
     'eHire',
     'Srinav Inc.',
     'Human Care Systems',
     'NewAgeSys, Inc.',
     'PeerSource',
     'Citadel Information Services Inc',
     'Next Level Business Services, Inc.',
     'Solomons International',
     'Ventures Unlimited',
     'Paint Nite',
     'Ramy InfoTech',
     'Lime Brokerage, LLC',
     'Peningo Systems, Inc.',
     'BlueAlly, LLC',
     'Makro Technologies Inc',
     'ITRays Incorporation',
     'MW Partners LLC',
     'APN Software Services, Inc',
     'Imetris Corporation',
     'TSP',
     'Electronics For Imaging',
     'WEX Inc',
     'Cilver Technologies',
     'Daley & Associates, LLC',
     'CNET Global Solutions, INC',
     'Concept Software & Services, Inc.',
     'BrickLogix',
     'Unique System Skills LLC',
     'IBA Software Technologies',
     'XSell Resources',
     'Systems Recruiters, LLC',
     'TECH Projects',
     'Continental Resources Inc.',
     'Tekizma',
     'Hire IT People',
     'Ironside',
     'Eveear Tech',
     'Sapvix',
     'ReVision Technologies Inc.',
     'Pantar Solutions, Inc.',
     'Systematix Technology Consultants Inc',
     'Visionaire Partners',
     'The Chartis Group',
     'Tanisha Systems, Inc.',
     'Cambridge Associates, LLC',
     'I.T. Software Solutions',
     'BridgeView IT',
     'TCA Consulting Group, Inc.',
     'STA',
     'Invictus Infotech',
     'Signature Technology Group',
     'Backyard Discovery',
     'MVP Consulting',
     'IMK Consulting',
     'SR International Inc.',
     'Pro Source Inc.',
     'Amtex Enterprises',
     'Rhode Island Quality Institute',
     'Datamatics Global Services Ltd.',
     'Prowess Consulting, Llc',
     'Apex 2000',
     'Zume IT, Inc.',
     'Sound Business Solutions',
     'Fabergent',
     'Stellar Soft Solutions Inc',
     'Hobbs Madison',
     'Qlarion',
     'SRI Infotech',
     'ITC Infotech',
     'Boston Services',
     'GCOM Software, Inc.',
     'The Brixton Group',
     'Yardi Systems',
     'Oxford International',
     'Global Data Management Inc',
     '360 IT Professionals Inc',
     'Marine Biological Laboratory',
     'OnX Enterprise Solutions',
     'Systems America, Inc',
     'Schireson Associates',
     'Techorbit, Inc',
     'AVA Consulting',
     'Lighthouse Computer Services',
     'Putnam Investments',
     'Covanex, Inc.',
     'Ascentiant International',
     'Niit Technologies, Inc.',
     'Dominion Enterprises',
     'Nous Infosystems',
     'Vectorsoft',
     'Nueva Solutions, Inc.',
     'APCON, Inc.',
     'Nantucket Island Resorts',
     'Samiti Technology Inc.',
     'Marshwinds International Inc.',
     'Amzur Technologies, Inc.',
     'Microexcel',
     'Presidio LLC',
     'Global Data Consultants',
     'Harmer Consultants, Inc.',
     'Sample6, Inc.',
     'GreenLight Staffing Group',
     'Zoom Tech Corp',
     'Palnar',
     'ETCS Inc',
     'Kanhi Systems LLC',
     'Info Origin Inc.',
     'Presidio Networked Solutions Group, LLC',
     'Fast Switch, Ltd.',
     'Spar Information Systems',
     'Top Source International Inc.',
     'Veritis Group, Inc.',
     'Beechwood Computing Ltd',
     'Openmind Technologies',
     'SJC4 Solutions',
     'Technical Link',
     'Isabella Stewart Gardner Museum',
     'DATASYS CONSULTING & SOFTWARE INC',
     'iTech US, Inc.',
     'Nam Info Inc',
     'Contech Systems Online',
     'Healthcare IT Leaders',
     'Catapult Recruiting',
     'Sigma Group',
     'Amtex Systems Inc.',
     'Red River Computer Co., Inc.',
     'Tec-Link',
     'Coppertree Staffing',
     'East Point Systems, Inc.',
     'Sophus IT Solutions',
     'Progressive Technology Solutions',
     'Data Exel',
     'Calibro Corp',
     'ENEA',
     'EBSCO Information Services',
     'UST Global Inc',
     'BankNewport',
     'Object Technology Solutions, Inc.',
     'KBRwyle',
     'Front Four Group',
     'nfrastructure',
     'Indeed Infotech',
     'JoulestoWatts Business Solutions Pvt. Ltd.',
     'Vedainfo Inc.',
     'Crossfire Consulting Corp',
     'Rockland Trust Company',
     'TEKsystems, Inc.',
     'Comm-Works',
     'Veredus',
     'Kforce Inc.',
     'Opal Force Inc',
     'Modis',
     'Ciber',
     'Abacus Technical Services',
     'British Telecommunications Public Limited Company',
     'RSM US',
     'Sierra Business Solution LLC',
     'Oportun, Inc',
     'CroytenER',
     'GM Financial',
     'Gurnet Consulting',
     'Tallan, Inc.',
     'Zeva Technology',
     'Charter Global, Inc.',
     'Strategic IT Staffing',
     'The Judge Group',
     'Fannie Mae',
     'Modulant IT Staffing',
     'System Soft Technologies',
     'Cohesion Consulting LLC',
     'Micro Focus',
     'RedRiver Systems L.L.C.',
     'ConsultNet, LLC',
     'Technology Resources Inc',
     'ANR Consulting Group, Inc.',
     'Mamsys',
     'Calance US',
     'Hexaware Technologies, Inc',
     'Infosys',
     'Butler America',
     'Maxsys Solutions, LLC',
     'Genesis TechSystems',
     'GDH Consulting',
     'NetSource, Inc.',
     'The Resource Collaborative',
     'Jack Richman & Associates / JRA Consulting',
     'Experis',
     'Advantage',
     'PVK Corporation',
     'Infotek Consulting Services Inc.',
     'SOAL Technologies, LLC.',
     'NORTHROP GRUMMAN',
     'Charles Schwab',
     'Computech Corporation',
     'OUTSOURCE Consulting Services, Inc',
     'Beacon Hill Staffing Group, LLC',
     'ZNA Infotek',
     'Austin Fraser USA',
     '2020 Companies',
     'VDart, Inc.',
     'Alans Group',
     'Object Information Service',
     'Bayforce',
     'UTIS,Inc.',
     'Sullivan and Cogliano',
     'Pixentia Corporation',
     'Starpoint Solutions LLC',
     'Linium',
     'Radisys Corporation',
     'Midwest Consulting Group, Inc.',
     'Wise Men Consultants',
     'eDataForce consulting LLC',
     'Orion Systems Integrators Inc',
     'Superior Group',
     'Resource Management International Inc.',
     'CompuCom Systems',
     'Mastech',
     'Logan Data',
     '7-Eleven, Inc.',
     'Voluble Systems LLC',
     'SGS Consulting',
     'Airbus DS Communications, Inc',
     'ettain group',
     'Nigel Frank International',
     'Anderson Frank',
     'Clear2Pay Consulting Americas, Inc',
     'ServeSolid, Inc',
     'SimSTAFF',
     'Complete Staffing Solutions',
     'Mercury Systems, Inc.',
     'Techpeople.US, Inc',
     'Maven Companies',
     'Guidance Software, Inc.',
     'Loganbritton',
     'Experian Limited',
     'Strategic Software Solutions, Inc.',
     'Q Investments, L.P.',
     'Global Resource Management, Inc.',
     'Federal Reserve Bank',
     'InfoGroup',
     'Global Conductor',
     'YER USA, Inc.',
     'Watts Water Technologies',
     'Dimension Consulting',
     'FocuzMindz',
     'Transamerican Information Systems',
     'ABAL Technologies Inc',
     'Genzeon',
     'Allstate Insurance',
     'Global Search Agency, Inc',
     'PricewaterhouseCoopers LLC',
     'LeadThem Consulting',
     'SQA',
     'Vistaprint USA Inc.',
     'VIVA USA INC',
     'Spectrum IT Global INC',
     'Nsight',
     'SyApps',
     'Chickasaw Nation Industries',
     'RIIM',
     'J.E. Ranta Associates',
     'MRoads',
     'Crestron Electronics Inc',
     'Advansoft International, Inc.',
     'Infowave Systems',
     'DTSC, Inc.',
     'Amazech Solutions',
     'CoreTechs LLC',
     'Harvard Partners, LLP',
     'TeamPersona',
     'Matlen Silver',
     'Citi',
     'Login Consulting Services, Inc',
     'Oberon IT',
     '10525742',
     'Corporate & Technical Recruiters, Inc',
     'Progressive IT',
     'Systel,Inc.',
     'EPE Innovations',
     'OSLO SOLUTIONS LLC',
     'KPIT Infosystems, Inc.',
     'UVS Infotech',
     'Ventana Solutions',
     'Prudent Technologies and Consulting',
     'I-Link Solutions',
     'Fusion Solutions, Inc.',
     'Accunet Solutions Inc',
     'Level 3 Communications',
     'LEVERAGEncy, LLC',
     'CESUSA, INC.',
     'The Fountain Group',
     'Eagle Investment Systems',
     '10120555',
     'Salt Search',
     'KASTECH Software Solutions Group',
     'Examination Management Services, Inc',
     'Precision Systems',
     'Tier Two Services, Inc.',
     'Mason Frank International',
     'Intellisoft Technologies',
     'GHA Technologies',
     'OspynTech',
     'Green Key Resources',
     'MIT Information Services and Technology',
     'Biltmore Technologies',
     'Eros Technologies Inc.',
     'PMG Global',
     'Enterprise Consulting Services Inc.',
     'PDS Tech, Inc.',
     'OBOX Solutions',
     'Vaco - Dallas',
     '10112494',
     'PSG Global Solutions',
     'Datanomics',
     'West Coast Consulting LLC',
     'WinWire Technologies',
     'Mobilitie Management, LLC',
     'United Software Group',
     'Tcognition, Inc',
     'Systems Pros Inc.',
     'Asquare.com',
     'V.L.S. Systems, Inc',
     'Greenlight',
     'Cynet Systems',
     'Monroe Staffing Services',
     'Onward Technologies Inc.',
     'Business Information Services',
     'Randstad Technologies',
     'Walkwater Technologies',
     'Shimento, Inc.',
     'INTRATEK COMPUTER, INC.',
     'Synigent Technologies',
     'StaffLabs',
     'Texara Solutions',
     'Tech Mahindra (Americas) Inc.',
     'Principle Solutions Group',
     'Resource Spectrum',
     'SS Info Tech, Inc.',
     'RJT Compuquest',
     'ENTERNET BUSINESS SYSTEMS',
     '[24]7 Inc.',
     'Ubertal',
     'Intelliswift Software Inc',
     'Maintec Technologies Inc',
     'Tranzeal, Inc.',
     'NTT DATA Consulting',
     'Zaspar Technologies',
     'CRG',
     'Liberty Hardware Mfg Corporation',
     'Metro Systems Inc',
     'EisnerAmper',
     'The New York Times Company',
     'Mount Sinai Medical Center',
     'Joseph Harry Ltd',
     'IT-Talent a Division of Tech Brains',
     'Hyatt Leader',
     'Millennium Infotech',
     'Techlink, Inc.',
     'Reveille Technologies',
     'Resource Search Company',
     'Johnson Service Group, Inc.',
     'SANS Consulting Services, Inc',
     'Saven Technologies',
     'Career Developers',
     'ADP Automatic Data Processing, Inc.',
     'Nutech Systems, Inc.',
     'ACA Technology LLC.',
     'e-Primary',
     'Bridget Morgan llc',
     'Canopy One Solutions Inc',
     'Sierra Infosys Inc.',
     'Patel Consultants Corp',
     'Telecomm Software',
     'Fulcrum Worldwide',
     'Bond Street Group',
     'Mitchell Martin, Inc.',
     'Techno Staffing Inc.',
     'firstPRO, Inc.',
     'NewConfig LLC',
     'Vaktech',
     'IRIS Software, Inc.',
     'GFK Custom Research-North America',
     'DSC Resources, Inc.',
     'Adept Solutions Inc',
     'IT America',
     'VDX, Inc.',
     'Radiant System, Inc',
     'The Week Publications, Inc.',
     'ORS Partners, LLC',
     'ITDirectives',
     'Computer Consultants Interchange',
     'Informatic Technologies',
     'AETEA Information Technology Inc',
     'Dataformix',
     'AKVARR',
     'Protech Solutions Inc',
     'Birlasoft',
     'Eden Technologies',
     'Marcum Search LLC',
     'Next Step Staffing',
     'Trillium Solutions Group, Inc.',
     'The Atlantic Group',
     'Advantex Professional Services',
     'Sierra-Cedar, Inc.',
     'Sasken Communication Technologies',
     'Mindteck',
     'Blackstone Professional Recruiting',
     'SageOne Inc',
     'EPAM Systems',
     'Mana Products',
     'Apps IT LTD',
     'CosaTech, Inc.',
     'Access Staffing',
     'Technical Resource Network',
     'New York Technology Partners',
     'Combined Computer Resources',
     'Lloyd Information Technology',
     'Auburn Technical Svcs Group',
     'Elite Imaging',
     'ActioNet',
     'Enterprise Software Solutions, Inc',
     'Data Resource Technologies',
     '314E Corporation',
     'Alagen',
     'Enterprise Consulting Services',
     'Power IT Consultancy Inc',
     'The Denzel Group',
     'OneAPPS',
     'Mizuho Bank',
     'TAKUMI',
     'KHOJ Information Technology, Inc.',
     'Auritas',
     'NYC Dept of Info Technology (DoITT)',
     'Codigent',
     'Kalpathy Consulting Group LLC',
     'Synechron Inc.',
     'Samsung SDS America Inc',
     'NextGen Global Resources LLC',
     'Turesol LLC a Div. of Tunnell Consulting',
     'JSL Computer Services, Inc.',
     'Euclid Technologies',
     'Armstrong Consulting Group',
     'Superpac, Inc.',
     'RennerBrown',
     'CoreHive Computing LLC',
     'MG Software Consultants Inc',
     'Bhrigus',
     'Infinity Consulting Solutions',
     'Bicitis Group',
     'UNIVERSAL Technologies',
     'Norgate Technology Inc',
     'KPI Partners, Inc.',
     'InSource, Inc.',
     'Tower Management',
     'MTM Technologies',
     'MDT Technical',
     'Diverse Lynx Llc',
     'SAI Systems International, Inc.',
     'ConsultAdd',
     'Open Systems Technologies',
     'United Business Solutions Inc',
     'Netrovert Software Inc.',
     'Paramount Software Solutions, Inc',
     'Perennial Resources International',
     'HBC',
     'Data Intelligence LLC.',
     'NCSA Sports',
     'Global Associates',
     'Bon Consulting, Inc',
     'Pioneer IT Associates',
     'Talent Hub 360',
     'Caliber Business Systems',
     'Cognition Systems',
     'Mitaja Corp',
     'Dovetail Systems',
     'Ngusoft Inc',
     'Wiley',
     'iconectiv',
     'Capgemini',
     'Intelikore Corporation',
     'C&G Consulting Services',
     'Compworldwide',
     'SwitchLane Inc.',
     'Columbia University',
     '24 Seven, Inc.',
     'Matlen Silver Group, Inc.',
     'TechLink Systems, Inc.',
     'ISL Techsolutions Inc',
     'Symbioun Technologies, Inc',
     'Spectra Group',
     'National Board of Medical Examiners',
     'Source Mantra Inc',
     'Galmont Consulting',
     'SVK Technology Solutions',
     'Affinity Resource Group (MRI)',
     'Blue Horizon Tek Solutions, Inc.',
     'Datto Inc',
     'Intelligent Capital Network, Inc.',
     'SigmaCare',
     'Trivision Group, Inc.',
     'The Forum Group',
     'Estuate Inc.',
     'Ezprohub LLC',
     'Alltech Consulting Services, Inc.',
     'RTP Technology Corporation',
     'Ajulia Executive Search',
     'TechTalent Squared',
     'WPG Consulting, LLC',
     'Vision 3000 IT Business Solutions',
     'Pros2Plan',
     'DVI Technologies, Inc.',
     'Creative Solutions Services',
     'NetCom Systems, Inc.',
     'Career Karma Inc',
     'GMS Advisors',
     'Creative Circle, LLC',
     'Knack Systems LLC',
     'Netrocon Inc',
     'Gravitas Recruitment Group',
     'Bank Of America',
     'Syncro Technology Corp',
     'IntePros Consulting',
     'COESYS Solutions Inc',
     'The NPD Group',
     'S-Square Technologies',
     'Prescientq',
     'Castle Consulting',
     'Spring Lake Consulting',
     'Vandis Inc.',
     'Custom Staffing',
     'Pyramid Consulting, Inc.',
     'Pantheon',
     'Conde Group Inc.',
     'IT Works Recruitment Inc',
     'Technomax LLC',
     'DC&M Partners',
     'Prestige Staffing',
     'Davinci Tek',
     'Horizon International',
     'Infusion',
     'Harken Data Inc.',
     'Data Incorporated',
     'Associated Press',
     'Archer I.T., LLC',
     'Sterling Medical Devices',
     'Jean Martin, Inc',
     'Technology Resource Management',
     'Nihaki Systems, Inc.',
     'Yablon & Associates',
     'GM4 Recruitment Associates',
     'Chronos Global Inc.',
     'Sofia Technology',
     'Resolve Tech Solutions (RTS)',
     'BigBevy Consulting',
     'Career Management Associates',
     'CSG',
     'Project One, Inc.',
     'Acnovate',
     'McKean Defense Group',
     'RJS Associates',
     'TeleQuest Communications Inc.',
     'Ascend It Staffing',
     'Congruent Info-Tech',
     'Columbia IS Consulting Group',
     'Enclara Pharmacia',
     'WizSolution LLC',
     'Hatstand US, Inc.',
     'Smart Source Technologies',
     'BellSoft',
     'MAK Technologies, LLC',
     'New York University',
     'iSam Global, Inc.',
     'OpenSky Corp.',
     'Indotronix International Corp',
     'CCM Consulting Services',
     'HALLMARK GLOBAL TECHNOLOGIES INC',
     'Intellect Technologies',
     'Project Consulting Specialists',
     'Element Technologies Inc.',
     'Synerzy Software Solutions Inc',
     'CPS Recruitment',
     'Synkriom',
     'Tanu Infotech Inc',
     ...]



### *List Names of all the Columns in a Dataset*

In scenarios where one is not familiar with the dataset before hand, it proves very helpful to look at the headers of the columns in the dataset. This is especially true in cases where the breadth of the dataset is high.

Another scernario where listing the column names is useful is when a new dataset needs to be created with only a subset of the columns from the original dataset. In such a situation one can directly use the resulting list of column names and filter data by columns.

Below however, I'm only going to list the names of the columns using the function ```list()``` and providing the argument for the funtion with ```DataFrame.columns.values``` method.


```python
#list of columns in a dataset
list(df.columns.values)
```




    ['advertiserurl',
     'company',
     'employmenttype_jobstatus',
     'jobdescription',
     'jobid',
     'joblocation_address',
     'jobtitle',
     'postdate',
     'shift',
     'site_name',
     'skills',
     'uniq_id']



### *List Categories and Count Instances of each Category*

In datasets with categorical columns, determining the distribution of those categories is one of the first steps. One may also want to see if the categories are clean or not i.e. if one category is represented in different forms and how severe is the issue. For example, in this dataset 'Full Time' is represented in multiple forms such as 'Full-time', 'Fulltime', 'FULLTIME' and so on.

This can be quite easily done in python using the ```value_counts()``` functions. Below is an example of the same for the 'employmenttype_jobstatus column. The resulting output is a tabulated format. This output can also be used to report the distribution as a bar graph using matplotlib library.


```python
#count of instances for each value/category in a column
df['employmenttype_jobstatus'].value_counts()
```




    Full Time                                                                                                                              6734
    Contract W2                                                                                                                            1102
    Contract Corp-To-Corp, Contract Independent, Contract W2                                                                                629
    Full Time, Full Time                                                                                                                    617
    Contract Corp-To-Corp, Contract Independent, Contract W2, C2H Corp-To-Corp, C2H Independent, C2H W2                                     479
    Full Time, Permanent                                                                                                                    367
    Full Time, Full-time, Employee                                                                                                          331
    C2H W2                                                                                                                                  302
    Contract Corp-To-Corp                                                                                                                   265
    Unknown                                                                                                                                 230
    Contract Corp-To-Corp, Contract W2                                                                                                      206
    Full Time, Contract Corp-To-Corp, Contract Independent, Contract W2, C2H Corp-To-Corp, C2H Independent, C2H W2                          182
    Contract Independent, Contract W2                                                                                                       178
    Full Time, Perm                                                                                                                         177
    Full Time, Full-time                                                                                                                    156
    Full Time, Contract Corp-To-Corp, Contract Independent, Contract W2, C2H Corp-To-Corp, C2H Independent, C2H W2, Part Time               149
    C2H Corp-To-Corp, C2H Independent, C2H W2                                                                                               118
    Contract W2, Temp                                                                                                                       117
    Full Time, Contract W2                                                                                                                  113
    Full Time, Temporary                                                                                                                    110
    Full Time, Fulltime                                                                                                                     106
    Full Time, FTE                                                                                                                           99
    Contract W2, 12 months                                                                                                                   98
    Full Time, Contract Independent, Contract W2, C2H Independent, C2H W2                                                                    93
    Market related                                                                                                                           90
    Full Time, Direct Hire                                                                                                                   89
    Full Time, FULLTIME                                                                                                                      88
    Contract Corp-To-Corp, Contract Independent, Contract W2, 6+ Months                                                                      85
    Contract W2, 6 Months                                                                                                                    82
    Contract Independent, Contract W2, C2H Independent, C2H W2                                                                               77
                                                                                                                                           ... 
    Contract Corp-To-Corp, Contract W2, 24 months +                                                                                           1
    Contract W2, 12+months                                                                                                                    1
    Contract Corp-To-Corp, Contract Independent, Contract W2, 12 - 36 months                                                                  1
    Contract Corp-To-Corp, Contract Independent, Part Time, 6 Months+                                                                         1
    C2H W2, 6 month to perm                                                                                                                   1
    Full Time, Contract Corp-To-Corp, Contract Independent, Contract W2, contarct/fulltime                                                    1
    Full Time, Contract W2, 3 years                                                                                                           1
    open                                                                                                                                      1
    Full Time, Contract Independent, Contract W2, C2H Independent, C2H W2, 20 mo + renewals                                                   1
    C2H Independent, C2H W2, 6+Months                                                                                                         1
    $90k - 100k per year                                                                                                                      1
    C2H Corp-To-Corp, 6 Months Plus                                                                                                           1
    Full Time, Contract Corp-To-Corp, Contract W2, C2H Independent, Long term                                                                 1
    Contract W2, Part Time                                                                                                                    1
    Full Time, Contract Corp-To-Corp, Contract Independent, Contract W2, C2H Corp-To-Corp, C2H Independent, 1 year +                          1
    Full Time, Contract Corp-To-Corp, Contract Independent, Contract W2, C2H Corp-To-Corp, C2H Independent, C2H W2, Part Time, CTH            1
    Full Time, 6+                                                                                                                             1
    Contract Independent, annually renewing                                                                                                   1
    C2H W2, 3 month C-H                                                                                                                       1
    Full Time, Contract Corp-To-Corp, Contract Independent, Contract W2, 12 months +                                                          1
    Contract W2, ASAP                                                                                                                         1
    Contract Corp-To-Corp, Contract W2, 7 months                                                                                              1
    Full Time, C2H Corp-To-Corp, C2H Independent, C2H W2, 3 month CTH                                                                         1
    Contract Corp-To-Corp, Contract Independent, Contract W2, C2H Corp-To-Corp, C2H Independent, C2H W2, 3 Months Contract to                 1
    Contract Corp-To-Corp, Contract Independent, Contract W2, Duration is estimated to be 6 to 12 months.                                     1
    Full Time, Contract W2, 11 Months                                                                                                         1
    Contract Corp-To-Corp, Contract Independent, Contract W2, 06 months                                                                       1
    Contract W2, 12 months extendable                                                                                                         1
    Contract Corp-To-Corp, Contract Independent, Contract W2, 6 month maybe longer                                                            1
    Full Time, Contract Corp-To-Corp, Contract Independent, Contract W2, C2H Corp-To-Corp, C2H Independent, C2H W2, 6-12 Month Contract       1
    Name: employmenttype_jobstatus, Length: 2928, dtype: int64



### *Filter Rows Based on Condition*

Filtering a dataset based on a certain condition(s) is quite often useful both to drill down into the dataset and for building statistical model(s). One may however have one or more conditions and that is not an issue because either of them is quite easy and intutive to perform in Pandas.

The first example below shows filtering on a single condition. In this example, I've filtered out those rows where the column 'employmenttype_jobstatus' is not 'Full Time'. The way this works is, the outer most ```DataFrame[]``` is called and within that only one of the series is selected to filter on and the value to look for is provided i.e. 'Full Time' in this case. 


```python
#Filtering the dataset for just Full Time jobs and saving it as new dataset called 'df_FT'

df_FT = df[df['employmenttype_jobstatus']=='Full Time']
```

The dataset that remains after filtering out the values not required is assigned to a variable 'df_FT'. Below, I've listed the unique values in 'employmenttype_jobstatus' column in the df_FT dataframe. As the output shows, the only value in this columns is 'Full Time' as we required.


```python
df_FT['employmenttype_jobstatus'].unique()
```




    array(['Full Time'], dtype=object)



As mentioned above, it also possible to filter a dataset based on multiple conditions. The way to do this is similar to performing single condition filtering with a very small difference.

Below, I've filtered the original dataframe 'df', selecting only those rows where the 'employmenttype_jobstatus' is 'Full Time' and also where the 'company' is 'PWC'. 

In order to pass two or more conditions, the original dataframe is called and the coditions are mentioned inside the square brackets the same way as with single condition. However, the difference is, each of those conditions have to be enclosed in parentheses. Also, you have to mention how those multiple conditions have to be fulfilled i.e. should all the condition be satified or one just of them or a combition of the two. In other words, considering the example below, since I wanted both the conditions to be satisfied I used '&' to represent 'and' between the two conditions. To represent 'or' a single pipe symbol, '\|' needs to be used instead of '&'.


```python
#Filter dataset with multiple conditions. Filtering full-time PWC jobs

df_PWC_FT = df[(df['employmenttype_jobstatus']=='Full Time') & (df['company']=='PWC')]
```

As can be seen below, the only value in 'employmenttype_jobstatus' columns of the new dataset 'df_PWC_FT' is 'Full Time' and the only value in the 'company' column of this dataset is 'PWC' as was my requirement.


```python
df_PWC_FT['employmenttype_jobstatus'].unique()
```




    array(['Full Time'], dtype=object)




```python
df_PWC_FT['company'].unique()
```




    array(['PWC'], dtype=object)




```python
len(df_PWC_FT.index)
```




    2



I will end this post here. As mentioned at the beginning, these are some of the most common data transformations but these only form a very small subset of all transformations that can be done in Pandas. I will continue to write posts such as this explaining other transformation techniques.
