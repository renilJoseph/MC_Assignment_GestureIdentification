# MC_Assignment_GestureIdentification

## Pre-Requesites
 - Flask
 - Python3
 - Run command `pip install -r requirements.txt`
 

## Data
 * Data set: 106 subjects, 3 trails of 2 min per subject. 
 * There is the sampling rate variable in data-set which is 160. The raw data variable is a 3D matrix 106*3*19200. On each of the three trail there is 120 sec of signal with 160 Hz frequency which means we will have 19200 values.  
 * To read data:
    ```
    import scipy.io
    mat = scipy.io.loadmat('EEGDataset1.mat')
    mat['Raw_Data'].shape : a nd array of shape (106, 3, 19200)
    ```     
    
## Starting FOG Server.
 * Copy contents of folder 'project' into any computer which is connected to same network as the mobile.
 * On the system in which contents are copied, Inside 'server/main.py' file, update 'host' parameter(passed to app.run()) to the ip address of the current system.
 * Run `python main.py` from inside the folder 'server'. This will start the fog server.
 * Update the system ip address in the source code of android app.
 
## Running Cloud Server
 * Server is already running inside GCP App Engine.
 * Hitting from mobile app will work.

## Running Android app:
 * Build and install App from Android Studio
