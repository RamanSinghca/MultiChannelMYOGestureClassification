There are 3 python modules and 14 Training data csv files ,  1 Excel sheet for final results.

1. EMGFeatureExtraction.py: It contains all the methods to calculated various time domain features from EMG signal . 
2. GestureFeatureEngine.py: This module takes sample files in a folder and converts 12000 files into one csv file containing training data for classifiers.
3. GestureClassifier: Takes the  training CSV file and trains various models defined in the module. 

4.Some of the Training Data generated from various feature engineering approaches are also provided in the form of CSV files.
Following are the list of file provided: Each row in a file is a row representation of [50x8] input sample. 

Autocorrelation Features:
	autoCorrTrain_Lag_3.csv
	autoCorrTrain_Lag_5.csv
	autoCorrTrain_Lag_10.csv
	autoCorrTrain_Lag_20.csv
	autoCorrTrain_Lag_30.csv
	autoCorrTrain_Lag_40.csv
	autoCorrTrain_Lag_50.csv

Concordance Correlation Features:
	concCorrTrain_Lag_0.csv
	concCorrTrain_Lag_5.csv
	concCorrTrain_Lag_10.csv

Cross Correlation Features:
	crossCorrTrain_Lag_0.csv
	crossCorrTrain_Lag_5.csv

Energy Ratio Features:
	eneryRatioTrain_Lag_0.csv

Other Time Domain Features:
	otherFeatureTrain.csv
	
