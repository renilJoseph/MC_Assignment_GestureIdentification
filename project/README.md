# MC_Assignment_GestureIdentification

## Pre-Requesites
 - Flask
 - Python3
 

## Data
 * Data set: 106 subjects, 3 trails of 2 min per subject. 
 * There is the sampling rate variable in data-set which is 160. The raw data variable is a 3D matrix 106*3*19200. On each of the three trail there is 120 sec of signal with 160 Hz frequency which means we will have 19200 values.  
 * To read data:
    ```
    import scipy.io
    mat = scipy.io.loadmat('EEGDataset1.mat')
    mat['Raw_Data'].shape : a nd array of shape (106, 3, 19200)
    ```
    
     

## Execution (Local Testing)


## Working:
