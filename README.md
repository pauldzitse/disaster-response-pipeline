<li><a id='installing'></a>
Installing

To clone this GIT repository use:

git clone https://github.com/pauldzitse/disaster-response-pipeline


a href='executing'></a>
Program Executing

- To run ETL pipeline that cleans data and stores in database
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- To run ML pipeline that trains classifier and saves
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
- Run the following command in the app's directory to run your web app: python run.py

- Go to http://0.0.0.0:3001/

 
<li><a id='authors'></a>
## 3. Authors  

[Paul Dzitse](https://github.com/pauldzitse)

  
<li><a id='license'></a>
## 4. License

<li><a id='acknowledgement'></a>
## 5. Acknowledgement
  
  
Thanks to [Udacity](https://www.udacity.com/school-of-data-science) for providing such an excellent Data Science Nanodegree Program
    
Also big thanks to [appen](https://appen.com/) for providing messages dataset to train my model
    
<li><a id='screenshots'></a>
## 6. Screenshots
