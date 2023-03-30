here you can see a jupyter notebook as well as a python script, which basically do the same.
depending on your type of computer, we would recommend to use the notebook, since the script may error due to computational limits. the notebooks saves the data and labels in between.


prepare_texas_data.ipynb/generate_texas_data.py was used to crop and resize the texas depth images, to generate the corresponding labels and finally apply some data augmentation. output is train and test images and their labels and will be saved as .npy arrays into /DATA_npy

**!!! make sure you put the face images of the Texas database (at least the folder /PreprocessedImages which you get from https://live.ece.utexas.edu/research/texas3dfr/) inside DATA_depth_images_raw/opensource_DATA/texas/ !!!** the path has to match "../DATA_depth_images_raw/opensource_DATA/texas/PreprocessedImages" inside the python script *generate_texas_data.py*

there is a notebook which does basically the same for the BDR images, but unfortunately we cannot publish this open source

____________________________________________________

execute the jupyter notebook to generate the train and test data and their corresponding labels, they will be saved as .npy arrays into /DATA_npy

when running it the first time, it is very likely, that an error appears: "No module named 'keras.engine.topology'" 
look into the Readme.md inside the root to fix this 
________________________________________________

texas_info.csv is used to generate the labels for the texas data and is obtained from https://live.ece.utexas.edu/research/texas3dfr/ as well
_______________________________________________

texas.png can be generated with the jupyter notebook
