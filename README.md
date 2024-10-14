# Azure ML - Internal Project

<a data-flickr-embed="true" href="https://www.flickr.com/photos/198784222@N05/53138533225/in/dateposted-public/" title="Group 29 (2)"><img src="https://live.staticflickr.com/65535/53138533225_1abf5983d0_h.jpg" width="1600" height="400" alt="Group 29 (2)"/></a><script async src="//embedr.flickr.com/assets/client-code.js" charset="utf-8"></script>

## The internal project
The goal of this project is to prepare for the work with the Baramundi. We want to make the ML pipeline similar to theirs. 
Develop it locally and then move it to cloud computing platform - Azure Machine Learning.

### Structure
```bash
$ tree
.
│   
│   # Directory containing the entire dataset used within the pipeline
├── data
│  │
│  │ # Data aggregated into weekly files
│  ├── aggregated
│  │
│  │ # Data in its preprocessed format
│  ├── preprocessed
│  │  │
│  │  │ # Data used for training the model and pipeline development
│  │  ├── train
│  │  │
│  │  │ # Data used for automation development
│  │  ├── update_1
│  │  │
│  │  │ # Data used for final testing
│  │  └── update_2
│  │
│  │ # Data in its raw format without any preprocessing
│  └── raw
│
│   # Implementation of the consecutive steps from the pipeline graphic above
└── pipeline
   │
   │ # Script for aggregating the data into weekly files
   ├── aggregation
   │
   │ # Lime algorithm - creating visual insights into the model's decision-making process
   ├── model_explanations
   │
   │ # Scripts for preprocessing the data and spliting it into train and test sets
   ├── preprocessing
   │
   │ # Scripts to visualize data coming out of the encoders, reducing alghoritms and visualization of anomaly detection
   ├── evaluation
   │
   │ # Folder containing training scripts, models and hyperparameter tuning process
   └── training
      │
      │ # Hyperparameter tuning process and related scripts
      ├── hyperparameter_tuning
      │
      │ # Implementations of different machine learning models
      ├── models
      │
      │ # Directory for saving trained models
      ├── trained_models
      │
      │ # Training process and related scripts
      └── training
```



### Raw data
Download [this](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports) 
folder with [this](https://download-directory.github.io/) tool. Place all '.csv' files in `root/data/raw`.

### Preprocessing
The data processing focuses on removing unwanted features, removing corrupted rows, filling null values. In this step we
also perform simple aggregation, that on the file level joins country subregions into one. Run the 
`root/pipeline/preprocessing/preprocessing_script.py`. It will split files into three folders 'train', 'update_1', 'update_2'.
First one will be used for pipeline development, second one for automation development, third one will use as final test.

### Aggregation
In order to normalize data, we join daily records into week-size chunks. Sum of Confirmed, Deaths and median of Incident_Rate, Case_Fatality_ratio.

### Training
We train unsupervised model SDAE (Stochastic Denoising AutoEncoder). It is a specific neural network architecture used for
dimensionality reduction and anomaly detection. We will use it for both.

### Hyperparameter Tuning
This stage tests different architectures of SDAE neural network and different hyperparameter settings.
The process iterates over specified parameters and use them to train, evaluate and retrieve the best model.

### Evaluation
Evaluation process contains anomaly detection and visualization. We use previously trained model to calculate the
mean square error of each record, then display time series of different features with marked outlier records.

### Interference
We use the best founded model to make 2D representation of records.

### Model Explainability
Trains LIME to analyse why model makes its prediction in such way.

### Plots - Processed data
Plot that for a few chosen countries shows the number of confirmed cases over time.

### Plots - Aggregated data
Plot that for a few chosen countries shows the number of confirmed cases over time.

### Plots - Result visualization
This should display for a certain day the clusters of countries that are most similar to one another
