# Spinal cord Segmentation projet

This folder contains all the elements to build a U-net segmenting the grey and white matter from MRI 3D images.
The Spinal_Cord_Segmentation.ipynb file contains and shows all the steps that are necessary to achieve segmentation. 
The Notebook format was chosen for its illustative capability and inline plotting.


The python functions used are gathered into the `./src/` folder:
    - pre_processing.py : contains all the functiosn used to reformat and apply a first pre-processing to the images 
    - data_generator.py : contains the DataGenerator class and other functions to slit and treat the data before feeding to the tensorflow model
    - unet_model.py : contains the code used to build the architecture as well as a custom metric to monitor training.
    - post_processing.py : contains the Metric class used to quantify the model's performance for the multiclassification task. Other functions are present to transorm the prediction. Finally the function necessary to characterise the segmentation are also in this file.
    
    
The `./images_db/` folder contains the images and the data stored as follow:
    images_db/
        |_ train/
            |_ images/
                |_ 2D-crop-128x128/
            |_ masks/
                |_ 2D-crop-128x128/
        |_ test/
            |_ images/
                |_ 2D-crop-128x128/
                
The data in the `2D-crop-128x128/` folders have the same naming to have an easier match between sample and label files.



The `./Data/` folder contains:
    - the saved best weights of the model during training 
    - the predictions in *.p files
    - the partitioning of the data in *.p file
    - the SpinalCharacteristics.csv file containing the extracted features from the segmentation
    
    
Author : Jean-Baptiste PROST - February 2020.# Spinal-Chord-Segmentation
