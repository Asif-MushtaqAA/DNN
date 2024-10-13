# DNN
An encoder-decoder CNN model capable of predicting flow field variables around an aerofoil.

The model predcits flow field variables (density, u velocity, v velocity, internal energy anf turbulence intensity) based on signed distance function (sdf) array of an aerofoil.

The model in implemented using PyTorch.

UI folder has 2 scripts to generate sdf arrays and then predict flow field using DNN. Run the console in the UI folder.
Visualiser script allows to visualise the inferred flow field. Run the console in the Visualiser folder.  
Example implementation can be found in the respective scripts.

Note- make sure to use correct paths and download the trained DNN model frm google drive.

