# AI NLP for depression detection 

This repository was created to detect depression indicators from natural speech with tf.keras. To achive this goal dataset scrapped from reddit with 7731 instances has been used(https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned). 
Web deployment has been built using django framework.
<br /><br />
Model achieved 92.3% accuracy on train dataset and 90.0% accuracy on test dataset.
<br /><br />
Example of prediction:
![ss](https://user-images.githubusercontent.com/68538575/189686676-9247cfeb-011d-4460-9af6-6ac6b44f3372.png)
<br /><br />
You can find original dataset in file data.csv and dataset after preprocessing and stemming in file stem_data.csv. <br /> My jupyter notebook is available in AI_NLP.ipynb file.
<br /> You can use Poetry tool to get all dependecies or find all requirements in pyproject.toml. <br />

To open website on your local server write:
<br /><br />
$ git clone https://github.com/LeweLotki/NLP_depression_detection.git
<br />
$ cd NLP_depression_detection/ML_deployment
<br />
$ python3 manage.py runserver
<br /><br />

To run training of the network write:
<br /><br />
$ git clone https://github.com/LeweLotki/NLP_depression_detection.git
<br />
$ cd NLP_depression_detection
<br />
$ python3 main.py
<br />
<br />
If you want to test diffrent sentences than default one, change content of sentence.txt file.
