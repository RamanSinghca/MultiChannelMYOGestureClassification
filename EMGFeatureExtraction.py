'''
Created on Oct 28, 2016

Calculates various Time Domain Features from a given Time series. 
These features are specifically suited for EMG signals and can be used for Gesture recognition task. 

@author: ramansingh, b4s79@unb.ca
'''
import math
import numpy as np
from scipy import signal



#Concordance Correlation  between x and y 
def concorln(x,y):
    """
    Calculates Lin's concordance correlation coefficient.
    Usage:   concorln(x,y)    where x, y are equal-length arrays
    Returns: concordance Correlation Coeffients between x and y 
"""
    covar = np.cov(x,y)[0][1]*(len(x)-1)/float(len(x))  # correct denom to n
    xvar = np.var(x)*(len(x)-1)/float(len(x))  # correct denom to n
    yvar = np.var(y)*(len(y)-1)/float(len(y))  # correct denom to n
    lincc = (2 * covar) / ((xvar+yvar) +((np.mean(x)-np.mean(y))**2))
    return lincc

#Lagged Concordance Cross Correlation
def lagConcordanceCorr(x,y,lag=None,verbose=False):
    '''Compute lag concordance correlations between 2 time series.
    '''
    if len(x)!=len(y):
        raise('Input variables of different lengths.')

    #--------Unify types of <lag>-------------
    if np.isscalar(lag):
        if abs(lag)>=len(x):
            raise('Maximum lag equal or larger than array.')
        if lag<0:
            lag=-np.arange(abs(lag)+1)
        elif lag==0:
            lag=[0,]
        else:
            lag=np.arange(lag+1)    
    elif lag is None:
        lag=[0,]
    else:
        lag=np.asarray(lag)

    #-------Loop over lags---------------------
    result=[]
    if verbose:
        print ('\n#<lagcorr>: Computing lagged-correlations at lags:',lag)

    for ii in lag:
        if ii<0:
            result.append(concorln(x[:ii],y[-ii:]))
        elif ii==0:
            result.append(concorln(x,y))
        elif ii>0:
            result.append(concorln(x[ii:],y[:-ii]))

    result=np.asarray(result)

    return result

#Lagged Cross Correlation
def lagcorr(x,y,lag=None,verbose=False):
    '''Compute lag correlations between 2 time series.
    '''
    if len(x)!=len(y):
        raise('Input variables of different lengths.')

    #--------Unify types of <lag>-------------
    if np.isscalar(lag):
        if abs(lag)>=len(x):
            raise('Maximum lag equal or larger than array.')
        if lag<0:
            lag=-np.arange(abs(lag)+1)
        elif lag==0:
            lag=[0,]
        else:
            lag=np.arange(lag+1)    
    elif lag is None:
        lag=[0,]
    else:
        lag=np.asarray(lag)

    #-------Loop over lags---------------------
    result=[]
    if verbose:
        print ('\n#<lagcorr>: Computing lagged-correlations at lags:',lag)

    for ii in lag:
        if ii<0:
            result.append(np.corrcoef(x[:ii],y[-ii:])[0, 1])
        elif ii==0:
            result.append(np.corrcoef(x,y)[0, 1])
        elif ii>0:
            result.append(np.corrcoef(x[ii:],y[:-ii])[0, 1])

    result=np.asarray(result)

    return result

#Energy of a time series Ei
def SSI(x):
    '''
    Energy of a time series Ei
    '''
    return np.dot(x, x)

#Energy Ratio between two time series
def energyRatioNormalized(x,y, E1):
    '''
    Energy Ratio between two time series
    '''
    Ei=SSI(x)
    Ej=SSI(y)
    if Ej !=0:
        return (Ei)/(Ej)
    else:
        return 0


# Measure of Trend   
def MTrend(x):
    y=signal.detrend(x)
    mt= 1-np.var(y)/np.var(x)
    return mt

# Integrated EMG
def IEMG(x):
    tempVal=0
    for i in x:
        tempVal=tempVal+np.absolute(i)
    return tempVal

# Mean Absolute Value
def MAV(x):
    tempVal=0
    N=len(x)
    for i in x:
        tempVal=tempVal+np.absolute(i)
    finalMAV=tempVal/N
    return finalMAV

#Modified Mean Absolute Value1
def MAV1(x):
    tempVal=0
    N= len(x)
    for i in x:
        if 0.25*N<=i<=0.75*N:
            tempVal=tempVal+np.absolute(i)
        else:
            tempVal=tempVal+0.5*np.absolute(i)
            
    finalMAV=tempVal/N
    return finalMAV

#Modified Mean Absolute Value2
def MAV2(x):
    tempVal=0
    N= len(x)
    for i in x:
        if 0.25*N<=i<=0.75*N:
            tempVal=tempVal+np.absolute(i)
        elif i<0.25*N:
            tempVal=tempVal+(4*np.absolute(i))/N
        else:
            tempVal=tempVal+(4*(np.absolute(i)-N))/N
            
            
    finalMAV=tempVal/N
    return finalMAV
        
#Variance of EMG
def VAR(x):
    tempVal=0
    N= len(x)
    for i in x:
        tempVal=tempVal+np.power(i, 2)
    finalVal=tempVal/(N-1)
    return finalVal

# Absolute Value of Third Temporal Moment
def TM3(x):
    tempVal=0
    N= len(x)
    for i in x:
        tempVal=tempVal+np.power(i, 3)
    finalVal=np.absolute(tempVal/(N))
    return finalVal

# Absolute Value of Fourth Temporal Moment
def TM4(x):
    tempVal=0
    N= len(x)
    for i in x:
        tempVal=tempVal+np.power(i, 4)
    finalVal=np.absolute(tempVal/(N))
    return finalVal

# Absolute Value of Fifth Temporal Moment
def TM5(x):
    tempVal=0
    N= len(x)
    for i in x:
        tempVal=tempVal+np.power(i, 5)
    finalVal=np.absolute(tempVal/(N))
    return finalVal

# Root Mean Square 
def RMS(x):
    tempVal=0
    N= len(x)
    for i in x:
        tempVal=tempVal+np.power(i, 2)
    finalVal=np.sqrt(tempVal/N).real
    return finalVal

#V-Order
def VOrder(x, v):
    tempVal=0
    N= len(x)
    for i in x:
        tempVal=tempVal+np.power(i, v)
    finalVal=np.power((tempVal/N), 1/v)
    return finalVal

# Log Detector
def LogDetector(x):
    tempVal=0
    N= len(x)
    for i in x:
        tempVal=tempVal+np.log10(np.absolute(i))
    finalVal=math.exp((tempVal/N))
    return finalVal

# Waveform Length
def WaveformLength(x):
    tempVal=0
    N= len(x)
    for i in range(N-1):
        currentEle=x[i]
        nextEle=x[i+1]
        tempVal=tempVal+np.absolute(nextEle-currentEle)
    finalVal=tempVal
    return finalVal

#Average Amplitude Change
def AAC(x):
    tempVal=0
    N= len(x)
    for i in range(N-1):
        currentEle=x[i]
        nextEle=x[i+1]
        tempVal=tempVal+np.absolute(nextEle-currentEle)
    finalVal=tempVal/N
    return finalVal

# Difference absolute standard Deviation value
def DASDV(x):
    tempVal=0
    N= len(x)
    for i in range(N-1):
        currentEle=x[i]
        nextEle=x[i+1]
        tempVal=tempVal+np.power((nextEle-currentEle),2)
    finalVal=math.sqrt(tempVal/(N-1)).real
    return finalVal

#Zero Crossing
def ZeroCrossing(x):
    tSignal= np.sign(x)  
    tSignal[tSignal==0] = -1     # replace zeros with -1  
    zero_crossings = np.where(np.diff(tSignal))[0]  
    return len(zero_crossings) 
 
#Myo pulse Percentage Rate 
def MYOP(x, threshold):
    tempVal=0
    N= len(x)
    for i in x:
        if i>=threshold:
            tempVal=tempVal+1
    finalVal=tempVal/N
    return finalVal

#Willison Amplitude
def WAMP(x, threshold):
    tempVal=0
    N= len(x)
    for i in range(N-1):
        currentEle=x[i]
        nextEle=x[i+1]
        fval=currentEle-nextEle
        if np.absolute(fval)>=threshold:
            tempVal=tempVal+1
    finalVal=tempVal
    return finalVal

#Slope Sign change
def SSC(x, threshold):
    tempVal=0
    N= len(x)
    for i in range(1,(N-1)):
        previousEle=x[i-1]
        currentEle=x[i]
        nextEle=x[i+1]
        fval=((currentEle-previousEle)*(currentEle-nextEle))
        if fval>=threshold:
            tempVal=tempVal+1
    finalVal=tempVal
    return finalVal

