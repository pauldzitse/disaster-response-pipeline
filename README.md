## Table of Contents

    <li><a href="#intro">1. Description
    <li><a href="#getting started">2. Getting Started
        <li><a href="#dependencies">Dependencies
        <li><a href="#installing">Installing
        <li><a href="#executing">Executing Program:
    <li><a href="#authors">3. Authors
    <li><a href="#License">4. License
    <li><a href="#acknowledgement">5. Acknowledgement
    <li><a href="#screenshots">6.Screenshots


 <a id='intro'></a>
  1. Description

This Project is one of the Data Science Nanodegree Program of [Udacity](https://www.udacity.com/school-of-data-science) in collaboration with  [appen](https://appen.com/). The initial dataset contains pre-labelled tweet and messages from real-life disaster situations. The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

    Processing Data, ETL Pipeline for extracting data from source, cleaning data and saveing them in a proper databse structure
    Machine Learning Pipeline for training a model to be able to classify text message in categories
    Web App tfor showing model results in real time.
      
<li><a href="#getting started">
## 2. Getting Started


  <li><a href="#dependencies">
    Dependencies

    Python 3.5+ (I used Python 3.7)
    Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
    Natural Language Process Libraries: NLTK
    SQLlite Database Libraqries: SQLalchemy
    Web App and Data Visualization: Flask, Plotly

   <li><a href="#installing">
    Installing

Clone this GIT repository:

git clone https://github.com/pauldzitse/disaster-response-pipeline

     <li><a href="#executing">
     Executing Program:

     Run the following commands in the project's root directory to set up your database and model.
        To run ETL pipeline that cleans data and stores in database, tpye: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        To run ML pipeline that trains classifier and saves, tpye: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

    Finally run the following command in the app's directory to run your web app: python run.py

    Go to http://0.0.0.0:3001/

<li><a href="#authors">
## 3. Authors

  [Paul Dzitse](https://github.com/pauldzitse)
  
<li><a href="#License">
## 4. License
   
  <li><a href="#acknowledgement">
## 5. Acknowledgement
    
    [Udacity](https://www.udacity.com/school-of-data-science) for providing such an excellent Data Science Nanodegree Program
    [appen](https://appen.com/) for providing messages dataset to train my model
    
    
<li><a href="#screenshots">
## 6.Screenshots
    
    

    
    
