#!/usr/bin/env python
# coding: utf-8

# In[49]:


#necessary libraries
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy
from scipy.stats import entropy
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit


# In[50]:


# setting data path for a directory with files
dataset_path_1st = 'C:/Users/Szymon/Desktop/pythoProject/1st_test'
dataset_path_2nd = 'C:/Users/Szymon/Desktop/pythoProject/2nd_test'
dataset_path_3rd = 'C:/Users/Szymon/Desktop/pythoProject/3rd_test/4th_test/txt'


# In[51]:


# Root Mean Squared Sum
def calculate_rms(df):
    result = []
    for col in df:
        r = np.sqrt((df[col]**2).sum() / len(df[col]))
        result.append(r)
    return result

# extract peak-to-peak features
def calculate_p2p(df):
    return np.array(df.max().abs() + df.min().abs())

# extract shannon entropy (cut signals to 500 bins)
def calculate_entropy(df):
    ent = []
    for col in df:
        ent.append(entropy(pd.cut(df[col], 500).value_counts()))
    return np.array(ent)

# extract clearence factor
def calculate_clearence(df):
    result = []
    for col in df:
        r = ((np.sqrt(df[col].abs())).sum() / len(df[col]))**2
        result.append(r)
    return result

def time_features(dataset_path, id_set=None):
    time_features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse']
    cols1 = ['B1_x','B1_y','B2_x','B2_y','B3_x','B3_y','B4_x','B4_y']
    cols2 = ['B1','B2','B3','B4']
    
    # initialize, creating parameters using function above according to set we take
    if id_set == 1:
        columns = [c+'_'+tf for c in cols1 for tf in time_features]
        data = pd.DataFrame(columns=columns)
    else:
        columns = [c+'_'+tf for c in cols2 for tf in time_features]
        data = pd.DataFrame(columns=columns)
        
    for filename in os.listdir(dataset_path):
        # read dataset
        raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep='\t')
        
        # time features
        mean_abs = np.array(raw_data.abs().mean())
        std = np.array(raw_data.std())
        skew = np.array(raw_data.skew())
        kurtosis = np.array(raw_data.kurtosis())
        entropy = calculate_entropy(raw_data)
        rms = np.array(calculate_rms(raw_data))
        max_abs = np.array(raw_data.abs().max())
        p2p = calculate_p2p(raw_data)
        crest = max_abs/rms
        clearence = np.array(calculate_clearence(raw_data))
        shape = rms / mean_abs
        impulse = max_abs / mean_abs
        
        if id_set == 1:
            mean_abs = pd.DataFrame(mean_abs.reshape(1,8), columns=[c+'_mean' for c in cols1])
            std = pd.DataFrame(std.reshape(1,8), columns=[c+'_std' for c in cols1])
            skew = pd.DataFrame(skew.reshape(1,8), columns=[c+'_skew' for c in cols1])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,8), columns=[c+'_kurtosis' for c in cols1])
            entropy = pd.DataFrame(entropy.reshape(1,8), columns=[c+'_entropy' for c in cols1])
            rms = pd.DataFrame(rms.reshape(1,8), columns=[c+'_rms' for c in cols1])
            max_abs = pd.DataFrame(max_abs.reshape(1,8), columns=[c+'_max' for c in cols1])
            p2p = pd.DataFrame(p2p.reshape(1,8), columns=[c+'_p2p' for c in cols1])
            crest = pd.DataFrame(crest.reshape(1,8), columns=[c+'_crest' for c in cols1])
            clearence = pd.DataFrame(clearence.reshape(1,8), columns=[c+'_clearence' for c in cols1])
            shape = pd.DataFrame(shape.reshape(1,8), columns=[c+'_shape' for c in cols1])
            impulse = pd.DataFrame(impulse.reshape(1,8), columns=[c+'_impulse' for c in cols1])
            
        else:
            mean_abs = pd.DataFrame(mean_abs.reshape(1,4), columns=[c+'_mean' for c in cols2])
            std = pd.DataFrame(std.reshape(1,4), columns=[c+'_std' for c in cols2])
            skew = pd.DataFrame(skew.reshape(1,4), columns=[c+'_skew' for c in cols2])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,4), columns=[c+'_kurtosis' for c in cols2])
            entropy = pd.DataFrame(entropy.reshape(1,4), columns=[c+'_entropy' for c in cols2])
            rms = pd.DataFrame(rms.reshape(1,4), columns=[c+'_rms' for c in cols2])
            max_abs = pd.DataFrame(max_abs.reshape(1,4), columns=[c+'_max' for c in cols2])
            p2p = pd.DataFrame(p2p.reshape(1,4), columns=[c+'_p2p' for c in cols2])
            crest = pd.DataFrame(crest.reshape(1,4), columns=[c+'_crest' for c in cols2])
            clearence = pd.DataFrame(clearence.reshape(1,4), columns=[c+'_clearence' for c in cols2])
            shape = pd.DataFrame(shape.reshape(1,4), columns=[c+'_shape' for c in cols2])
            impulse = pd.DataFrame(impulse.reshape(1,4), columns=[c+'_impulse' for c in cols2])
            
        mean_abs.index = [filename]
        std.index = [filename]
        skew.index = [filename]
        kurtosis.index = [filename]
        entropy.index = [filename]
        rms.index = [filename]
        max_abs.index = [filename]
        p2p.index = [filename]
        crest.index = [filename]
        clearence.index = [filename]
        shape.index = [filename]
        impulse.index = [filename] 
        
        # concat - merging data
        merge = pd.concat([mean_abs, std, skew, kurtosis, entropy, rms, max_abs, p2p,crest,clearence, shape, impulse], axis=1)
        data = data.append(merge)
        
    if id_set == 1:
        cols = [c+'_'+tf for c in cols1 for tf in time_features]
        data = data[cols]
    else:
        cols = [c+'_'+tf for c in cols2 for tf in time_features]
        data = data[cols]
        
    data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')
    data = data.sort_index()
    return data   


# In[52]:


#saving data 
set1 = time_features(dataset_path_1st, id_set=1)
set1.to_csv('set1_timefeatures.csv')


# In[53]:


#reading data from file
set1 = pd.read_csv("./set1_timefeatures.csv")
set1 = set1.rename(columns={'Unnamed: 0':'time'})
#set1 = set1.set_index('time')
last_cycle = int(len(set1))


# In[54]:


features = set1.copy()
# Our next step is to load the dataset and choose the algorithm of getting the result from defined functions above. 

#simple moving average SMA
ma = pd.DataFrame()
ma['B4_x_mean'] = features['B4_x_mean']
ma['SMA'] = ma['B4_x_mean'].rolling(window=5).mean()
ma['time'] = features['time']


# In[55]:


#Cumulative Moving Average
ma['CMA'] = ma["B4_x_mean"].expanding(min_periods=10).mean()


# In[56]:


#Exponantial Moving Average
ma['EMA'] = ma['B4_x_mean'].ewm(span=40,adjust=False).mean()


# In[57]:


ma.plot(x="time", y= ['B4_x_mean','SMA','CMA','EMA'])


# In[66]:


# time to failure
# from the base of given data it calculates PCA (Principal Component Analysis).
# If the data’s amplitude is relatively big, and we suspect it,
# then we can substitute the data with its mean function from specified span.
# After transforming it according to given dimensions of the data the explained_variance_ratio is printed.
# It is done for the information if the variance is correct. 
# Then we calculate and return the degradation state of bearing for each day.
def health_indicator(bearing_data,use_filter=False):    
    data = bearing_data.copy()
    if use_filter:
        for ft in data.columns:
            data[ft] = data[ft].ewm(span=40,adjust=False).mean()
    pca = PCA()
    X_pca = pca.fit_transform(data)
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    print("Variance for PC 1:"+str(pca.explained_variance_ratio_[0]))
    health_indicator = np.array(X_pca['PC1'])
    degredation = pd.DataFrame(health_indicator,columns=['PC1'])
    degredation['cycle'] = degredation.index
    degredation['PC1'] = degredation['PC1']-degredation['PC1'].min(axis=0)
    return degredation

# returns an array describing curve_fit (uses non-linear least squares to fit a function to a data).
#  In the x-axis there is cycle number, and y is degredation level. 
#  The function applied to the data is exponential (exp_fit – a*np.exp(abs(b)*x)) to present the number of cycle that fails. 

def fit_exp(df,base=500,print_parameters=False):
    x =np.array(df.cycle)
    x = x[-base:].copy()
    y = np.array(degredation.PC1)
    y = y[-base:].copy()
    def exp_fit(x,a,b):
        y = a*np.exp(abs(b)*x)
        return y
    #initial parameters affect the result
    fit = curve_fit(exp_fit,x,y,p0=[0.01,0.01],maxfev=1000)
    if print_parameters:
        print(fit)
    return fit

# As parameters takes degradation and curve_fit we created earlier.
# It takes exponential method described above to calculate prediction cycle of failure for bearing. 

def predict(X_df,p):
    x =np.array(X_df.cycle)
    a,b = p[0]
    fit_eq = a*np.exp(abs(b)*x)
    return fit_eq
log = [[],[]]


# In[67]:


#variable for incrementing index
prediction_cycle = 600
#variable for keeping intial value
init_cycle = prediction_cycle


# In[68]:


#selected_features = ['mean','std','kurtosis','skew','entropy',
#        'rms','max','p2p','crest','shape','impulse']
selected_features = ['max','p2p','rms']

bearing = 1
B_x = ["B{}_x_".format(bearing)+i for i in selected_features]
early_cycles = set1[B_x][:init_cycle]
early_cycles_pca = health_indicator(early_cycles,use_filter=True)


# In[69]:



while prediction_cycle<last_cycle:
    # run this again, again to simulate life-cycle of a bearing
    # We predict number of cycle that fail with some incrementation of cycles with help of functions described above.-
    # - for graph purpose
    # Additionally we predict its value by logarithmic method.
    # Value of prediction and actual value is presented on the plot.
    
    data = set1[B_x][:prediction_cycle]
    degredation = health_indicator(data,use_filter=True)
    fit = fit_exp(degredation,base=250)

    prediction = predict(degredation,fit)
    m,n = fit[0]
    thres = 2
    fail_cycle = (np.log(thres/m))/abs(n)
    log[0].append(prediction_cycle)
    log[1].append(fail_cycle)

    #print(m,n)
    print('Failed at'+str(fail_cycle))
    fig =plt.figure()
    ax =fig.subplots()
    ax.plot([0,prediction_cycle],[2,2])
    ax.set_title('Cycle: '+str(prediction_cycle))
    ax.scatter(degredation['cycle'],degredation['PC1'],color='b',s=5)
    ax.plot(degredation['cycle'],prediction,color='r',alpha=0.7)
    ax.legend(['threshold','prediction'])
    plt.show()
    increment_cycle = 25
    prediction_cycle += increment_cycle
    


# In[70]:


# In the last part the table with result is created. 
# In the last column there is an information if a trial is valid (prediction of cycle that fails is within the range of cycles).
d = {'time':set1['time'][init_cycle::increment_cycle],'cycle': log[0], 'prediction': (np.array(log[1]))}
df = pd.DataFrame(data=d)
df['IsValid'] = df['prediction']<last_cycle

print("For bearing "+str(bearing))
print(df)
print(df.tail(60))


# In[71]:


set2 = time_features(dataset_path_2nd, id_set=2)
set2.to_csv('set2_timefeatures.csv')


# In[72]:


set2 = pd.read_csv("./set2_timefeatures.csv")
set2 = set2.rename(columns={'Unnamed: 0':'time'})


# In[88]:


log = [[],[]]
#variable for incrementing index
prediction_cycle = 550
#variable for keeping intial value
init_cycle = prediction_cycle


# In[89]:


selected_features = ['max','p2p']
bearing = 3
B_x = ["B{}_".format(bearing)+i for i in selected_features]
early_cycles = set2[B_x][:init_cycle]
early_cycles_pca = health_indicator(early_cycles,use_filter=True)


# In[90]:


last_cycle = int(len(set2))
print(int(len(set2)))


# In[91]:


while prediction_cycle<last_cycle:
    data = set2[B_x][:prediction_cycle]
    degredation = health_indicator(data,use_filter=True)
    fit = fit_exp(degredation,base=250)

    prediction = predict(degredation,fit)
    m,n = fit[0]
    thres = 2
    fail_cycle = (np.log(thres/m))/abs(n)
    log[0].append(prediction_cycle)
    log[1].append(fail_cycle)

    #print(m,n)
    print('failed at'+str(fail_cycle))

    fig =plt.figure()
    ax =fig.subplots()
    ax.plot([0,prediction_cycle],[2,2])
    ax.set_title('Cycle: '+str(prediction_cycle))
    ax.scatter(degredation['cycle'],degredation['PC1'],color='b',s=5)
    ax.plot(degredation['cycle'],prediction,color='r',alpha=0.7)
    ax.legend(['threshold','prediction'])
    plt.show()
    increment_cycle = 25
    prediction_cycle += increment_cycle


# In[92]:


#True labels represent alerts which are given before real end cycle!
d = {'time':set2['time'][init_cycle::increment_cycle],'cycle': log[0], 'prediction': (np.array(log[1]))}
df = pd.DataFrame(data=d)
df['is valid'] = df['prediction']<last_cycle
df.head(30)


# In[ ]:




