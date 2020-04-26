
# coding: utf-8

# In[ ]:


import numpy as np                  # vectors and matrices
import pandas as pd                 # tables and data manipulations
import warnings                     # There would be no warnings anymore
warnings.filterwarnings('ignore')
import hydrofunctions as hf
import matplotlib.pyplot as plt
observation = hf.NWIS('03335500', 'iv', start_date='2019-01-01',end_date='2019-06-30')
observation.get_data()
Timeseries = observation.df()
Timeseries.columns = ["discharge", "flag"]
Timeseries.head()
Timeseries.to_csv("Timeseries.csv", sep=',')
Daily = Timeseries.resample('D').mean()


# In[ ]:


# Check the diachrge plot
get_ipython().run_line_magic('matplotlib', 'inline')
time = pd.to_datetime(Daily.index)
plt.plot(time, Daily.discharge)
plt.xlabel('Time')
plt.ylabel('Discharge (cfs)')
plt.title('Discharge Hydrograph')
plt.show()


# In[ ]:


# Building a first order exponential smoothing model
class Exp_Smoothing:
    """
    Exponential Smoothing model
    # series - initial time series
    # alpha - exponential smoothing parameter
    """
    
    def __init__(self, series, alpha):
        self.series = series
        self.alpha = alpha
        
    def exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        for i in range(len(self.series)):
            if i == 0: # components initialization
                smooth = self.series[0]
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                continue
            else:
                val_now = self.series[i - 1]
                val_pre = self.result[i - 1]
                smooth = self.alpha*val_now + (1-self.alpha)*val_pre
                self.result.append(smooth)
            self.Smooth.append(smooth)


# In[ ]:


model = Exp_Smoothing(Daily.discharge, 1)
model.exponential_smoothing()
result = model.result
print(result)


# In[ ]:


# Plot output from first order model
plt.plot(time, Daily.discharge, time, result)
plt.xlabel('Time')
plt.ylabel('Discharge (cfs)')
plt.title('Discharge Hydrograph')
plt.show()


# In[ ]:


# Define a class for prediction function
class Exp_Smoothing_Prediction:
    """
    Exponential Smoothing model
    # series - initial time series
    # alpha - exponential smoothing parameter
    """
    
    def __init__(self, series, alpha, n_preds):
        self.series = series
        self.alpha = alpha
        self.n_preds = n_preds
    def exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series[0]
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                continue
            if i >= len(self.series):
                val_now = self.result[i - 1]
                val_pre = self.result[i - 1]
                smooth = self.alpha*val_now + (1-self.alpha)*val_pre
                self.result.append(smooth)
            else:
                val_now = self.series[i - 1]
                val_pre = self.result[i - 1]
                smooth = self.alpha*val_now + (1-self.alpha)*val_pre
                self.result.append(smooth)
            self.Smooth.append(smooth)


# In[ ]:


model_p = Exp_Smoothing_Prediction(Daily.discharge, 0.8, 5)
model_p.exponential_smoothing()
result = model_p.result
print(result)


# In[ ]:


# Plot the discharge of observation and prediction
dates = pd.date_range('20190101', periods=len(result))
plt.plot(time, Daily.discharge, dates, result)
plt.xlabel('Time')
plt.ylabel('Discharge (cfs)')
plt.title('Discharge Hydrograph')
plt.show()


# In[ ]:


# Function of returning the error of prediction
from sklearn.metrics import mean_squared_error
def Train_Score(param, series, validsize, loss_function=mean_squared_error):
    '''
    return error
    param -parameter for optimization
    series -timeseries dataset
    validsize -size of validation dataset
    '''
    model_p = Exp_Smoothing_Prediction(series[:-validsize], param, validsize)
    model_p.exponential_smoothing()
    result = model_p.result[-validsize:]
    val = list(series[-validsize:])
    return (loss_function(result, val))


# In[ ]:


# Output the error
a1 = Train_Score(0.8, Daily.discharge, 5, loss_function=mean_squared_error)
print('The error of prediction for 5 more days is:'+ str(a1))

