## Table of Contents
<li><a href="#intro">1. Project Description
<li><a href="#files">2. Important folders and files
<li><a href="#getting started">3. To start
<li><a href="#authors">4. Authors
<li><a href="#license">5. License
<li><a href="#acknowledgement">6. Acknowledgement
<li><a href="#images">7. Images


<a id='intro'></a>
## 1. Description

This Project is one of the Data Science Nanodegree Program of [Udacity](https://www.udacity.com/school-of-data-science) in collaboration with  [appen](https://appen.com/). The initial dataset contains pre-labelled tweet and messages from real-life disaster situations. The aim of the project is to build a Natural Language Processing tool that categorize messages.
     

The Project is divided in the following Sections:

- Processing Data, ETL Pipeline for extracting data from source, cleaning data and saving them in a proper database structure
- Machine Learning Pipeline for training a model to be able to classify text message in categories
- Web App for showing model results in real time.
     
 <a id='files'></a>
 ## 2. Important folders and files
  - **process_data.py**: This python executable code takes as its input csv files(message data and message categories datasets), and then creates a SQL database
  - **train_classifier.py**: This code trains the ML model with the SQL data base
  - **ETL Pipeline Preparation.ipynb**: process_data.py development process
  - **ML Pipeline Preparation.ipynb**: train_classifier.py. development process
  - **data**: This folder contains sample messages and categories datasets in csv format.
  - **app**: contains the run.py to initiate the web app.

<a id='getting started'></a>
## 3. To start    
### Dependencies
 
 - Python 3.5+ (I used Python 3.7)
 - Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
 - Natural Language Process Libraries: NLTK
 - SQLlite Database Libraries: SQLalchemy
 - Web App and Data Visualization: Flask, Plotly
 
### Installing

 - To clone this GIT repository use: git clone https://github.com/pauldzitse/disaster-response-pipeline

 ### Program Executing

  - To run ETL pipeline that cleans data and stores in database:
     - **python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**
  - To run ML pipeline that trains classifier and saves:
      - **python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl**
  - Run the following command in the app's directory to run your web app: 
      - **python run.py**
  - Go to <http://0.0.0.0:3001/>

 
<a id='authors'></a>
## 4. Authors 
     
   - [Paul Dzitse](https://github.com/pauldzitse)

<a id='license'></a>
## 5. License

<a id='acknowledgement'></a>
## 6. Acknowledgement
  
  - Thanks to [Udacity](https://www.udacity.com/school-of-data-science) for providing such an excellent Data Science Nanodegree Program
  - Also big thanks to [appen](https://appen.com/) for providing messages dataset to train my model

<a id='images'></a>
## 7. Images
