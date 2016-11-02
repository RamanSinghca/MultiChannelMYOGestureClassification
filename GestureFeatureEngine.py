'''
Created on Oct 27, 2016

Feature Extraction module to convert gesture sample [50x8] into training file. 

@author: ramansingh, b4s79@unb.ca
'''
import os
import csv
import pandas
import numpy as np
import statsmodels.tsa.stattools
from scipy.stats import kurtosis, skew 
from statsmodels.tsa.stattools import acf
import EMGFeatureExtraction as EMGFeatures
from sklearn.preprocessing import MinMaxScaler

# Path to under Home. Might be different for window based OS
userhome = os.path.expanduser('~')

# Input Data folder location
dataFolderPath=userhome+r'/Desktop/Data/thalmicMyoData/gestureData/'

# Training Data output folder
trainingOutLoc=userhome +r'/Desktop/Data/thalmicMyoData/trainingData/'

# File names for output data after feature extraction
acfTrainOut='autoCorrTrain_Lag_'
crossCorTrainOut='crossCorrTrain_Lag_'
concCorrTrainOut='concCorrTrain_Lag_'
eneryRatioTrainOut='eneryRatioTrain_Lag_'
otheFeatTrainOut='otherFeatureTrain'

def auto_correlation_feature_engine():
    acfLag=10

    finalArray=[]
    for gNo in range(1,7):
        for sampleNo in range(1,2001):
            gestureFile='Gesture{gNo}_Example{SampleNo}.txt'.format(gNo=gNo,SampleNo=sampleNo)
            Data = dataFolderPath+ gestureFile
            print('Processing: ', gestureFile)
            X = pandas.read_csv(Data, header=None);
            dataset = X.values
            arr = []

            for i in range (0, dataset.shape[1]):
                temp=[]
                xSeries= dataset[:,i]
                # Calculate AutoCorrelation Features
                acf=statsmodels.tsa.stattools.acf(xSeries, nlags=acfLag)  
                temp.append(acf[1:])
                arr.append(temp)
                
            arr=np.asarray(arr)                        
            svACF = np.linalg.svd(arr, full_matrices=True, compute_uv=False)
            finalIntanceRow=np.append(svACF, gNo)

            finalArray.append(finalIntanceRow)
            
    printToCSV(finalArray,acfTrainOut+'{lag}'.format(lag=acfLag))    

def cross_corr_feature_engine():
    nLag=0
    FinalOutput=[]
    for gNo in range(1,7):
        for sampleNo in range(1,2001):
            
            gestureFile='Gesture{gNo}_Example{SampleNo}.txt'.format(gNo=gNo,SampleNo=sampleNo)
            Data = dataFolderPath+ gestureFile
            print('Processing: ', gestureFile)
            
            X = pandas.read_csv(Data, header=None);
            dataset = X.values
            FinalIntance=[]
            for i in range (0, dataset.shape[1]-1):
                xSeries= dataset[:,i]
                for j in range (i+1, dataset.shape[1]):
                    ySeries= dataset[:,j]
                    correlationCoefA=EMGFeatures.lagcorr(xSeries, ySeries, lag=-nLag)[::-1] 
                    correlationCoefB=EMGFeatures.lagcorr(xSeries, ySeries, lag=nLag)
                    FinalIntance.extend(correlationCoefA)
                    if len(correlationCoefB)>1:
                        FinalIntance.extend(correlationCoefB[1:])
            FinalIntance.append(gNo)
            FinalOutput.append(FinalIntance) 
            
    printToCSV(FinalOutput,crossCorTrainOut+'{lag}'.format(lag=nLag))    

def concordance_corr_feature_engine():
    nLag=0
    FinalOutput=[]
    for gNo in range(1,7):
        for sampleNo in range(1,2001):
            
            gestureFile='Gesture{gNo}_Example{SampleNo}.txt'.format(gNo=gNo,SampleNo=sampleNo)
            Data = dataFolderPath+ gestureFile
            print('Processing: ', gestureFile)
            
            X = pandas.read_csv(Data, header=None);
            dataset = X.values
            FinalIntance=[]
            for i in range (0, dataset.shape[1]-1):
                xSeries= dataset[:,i]
                for j in range (i+1, dataset.shape[1]):
                    ySeries= dataset[:,j]
                    #print('CalculatingFor', i,j)
                    correlationCoefA=EMGFeatures.lagConcordanceCorr(xSeries, ySeries, lag=-nLag)[::-1] 
                    correlationCoefB=EMGFeatures.lagConcordanceCorr(xSeries, ySeries, lag=nLag)
                    FinalIntance.extend(correlationCoefA)
                    if len(correlationCoefB)>1:
                        FinalIntance.extend(correlationCoefB[1:])
            FinalIntance.append(gNo)
            FinalOutput.append(FinalIntance) 
            
    printToCSV(FinalOutput,concCorrTrainOut+'{lag}'.format(lag=nLag))  

def energy_ratio_feature_engine():
    nLag=0
    FinalOutput=[]
    for gNo in range(1,7):
        for sampleNo in range(1,2001):
            
            gestureFile='Gesture{gNo}_Example{SampleNo}.txt'.format(gNo=gNo,SampleNo=sampleNo)
            Data = dataFolderPath+ gestureFile
            print('Processing: ', gestureFile)
            
            X = pandas.read_csv(Data, header=None);
            dataset = X.values
            FinalIntance=[]
            E1=EMGFeatures.SSI(dataset[:,0]) #First Columns Energy
            for i in range (0, dataset.shape[1]-1):
                xSeries= dataset[:,i]
                for j in range (i+1, dataset.shape[1]):
                    ySeries= dataset[:,j]
                    #print('CalculatingFor', i,j)
                    energyRatioCoeff=EMGFeatures.energyRatioNormalized(xSeries, ySeries, E1)
                    FinalIntance.append(energyRatioCoeff)
            FinalIntance.append(gNo)
            FinalOutput.append(FinalIntance) 
            
    printToCSV(FinalOutput,eneryRatioTrainOut+'{lag}'.format(lag=nLag))    

def other_feature_engine():
    finalArray=[]
    for gNo in range(1,7):
        for sampleNo in range(1,2001):
    
            gestureFile='Gesture{gNo}_Example{SampleNo}.txt'.format(gNo=gNo,SampleNo=sampleNo)
            Data = dataFolderPath+ gestureFile
            print('Processing: ', gestureFile)
            
            X = pandas.read_csv(Data, header=None);
            dataset = X.values
            allOtherFeatures=[]
            
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            
            for i in range (0, dataset.shape[1]):
                tempOtherFeat=[]
                xSeries= dataset[:,i]

                # Calculate Features From Time Domain
                MTrend_R,kurtosis_R,skew_R,IEMG_R,MAV_R,MAV1_R,MAV2_R,SSI_R,VAR_R, TM3_R,TM4_R,TM5_R,RMS_R,VOrder_R,LogDetector_R,WaveformLength_R,AAC_R,DASDV_R,ZeroCrossing_R,MYOP_R,WAMP_R,SSC_R=EMGFeatures.MTrend(xSeries), kurtosis(xSeries), skew(xSeries), EMGFeatures.IEMG(xSeries), EMGFeatures.MAV(xSeries),EMGFeatures.MAV1(xSeries),  EMGFeatures.MAV2(xSeries), EMGFeatures.SSI(xSeries), EMGFeatures.VAR(xSeries), EMGFeatures.TM3(xSeries), EMGFeatures.TM4(xSeries), EMGFeatures.TM5(xSeries), EMGFeatures.RMS(xSeries), EMGFeatures.VOrder(xSeries, 2), EMGFeatures.LogDetector(xSeries), EMGFeatures.WaveformLength(xSeries), EMGFeatures.AAC(xSeries),  EMGFeatures.DASDV(xSeries), EMGFeatures.ZeroCrossing(xSeries), EMGFeatures.MYOP(xSeries, 50), EMGFeatures.WAMP(xSeries, 50), EMGFeatures.SSC(xSeries, 50)
                tempOtherFeat.extend(([MTrend_R,kurtosis_R,skew_R,IEMG_R,MAV_R,MAV1_R,MAV2_R,SSI_R,VAR_R, TM3_R,TM4_R,TM5_R,RMS_R,VOrder_R,LogDetector_R,WaveformLength_R,AAC_R,DASDV_R,ZeroCrossing_R,MYOP_R,WAMP_R,SSC_R]))
                
                allOtherFeatures.append(tempOtherFeat)                                
            allOtherFeatures=np.asarray(allOtherFeatures)
            
            svOthers= np.linalg.svd(allOtherFeatures, full_matrices=True, compute_uv=False)
            finalIntanceRow=np.append(svOthers , gNo)
            finalArray.append(finalIntanceRow)
            
    printToCSV(finalArray,otheFeatTrainOut)    

def printToCSV(finalArray, fileName):
    fl = open(trainingOutLoc+'{fn}.csv'.format(fn=fileName), 'w')
    writer = csv.writer(fl)
    for values in finalArray:
        writer.writerow(values)
    fl.close() 


'''
Uncomment one of them to run sample data [50x8] file and convert it into training data for classifier stage.
'''
#other_feature_engine()
#energy_ratio_feature_engine()
#concordance_corr_feature_engine()
#cross_corr_feature_engine()
#auto_correlation_feature_engine()


