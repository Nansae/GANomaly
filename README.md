# GANomaly
The repository has implemented the **GANomaly**   

**Requirements**
* python 3.6   
* tensorflow-gpu==1.14   
* pillow
* matplotlib

## Concept
![ganomaly_concept](https://user-images.githubusercontent.com/11286586/93835035-e981c780-fcb8-11ea-8053-5fd4f6e2bb17.png)


## Files and Directories
* config.py : A file that stores various parameters and path settings.
* model.py : GANomaly's network model file
* train.py : This file load the data and learning with GANomaly.
* utils.py : Various functions such as loading data 

## Train
1. You prepare the data.
- You can load the data by using the **read_images** function in the **utils.py**
- The structure of the data must be as follows:
   ```
   ROOT_FOLDER
      |   
      |--------SUBFOLDER (Class 0)   
      |          |------image1.jpg   
      |          |------image2.jpg   
      |          |------etc..   
      |--------SUBFOLDER (Class 1)   
      |          |------image1.jpg   
      |          |------image2.jpg   
      |          |------etc..
   ```

2. Run **train.py**
3. The result images are saved per epoch in the **sample_data**


## Reference
* Akcay, Samet, Amir Atapour-Abarghouei, and Toby P. Breckon. "Ganomaly: Semi-supervised anomaly detection via adversarial training." Asian conference on computer vision. Springer, Cham, 2018.
