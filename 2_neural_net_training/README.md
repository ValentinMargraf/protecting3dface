inside this folder you'll find several scripts for model training and prediction.
to properly execute those **make sure, that you have generated the .npy data by using the python script in  /1_data_preprocessing!**



## 1. 29_persons_relu.h5
the weights of the trained neural net (standard format of tensorflow)

## 2. several python scripts:

*predict.py*                      execute python3 predict.py in order to run this script. this will then load the TEST DATA from the folder 1_data_preprocessing/DATA_npy/, instantiate a resnet model, load the 29_persons_relu.h5 weights, generate the embeddings, check the accuracy, should be approx. 0.974, as stated also in the paper (check if threshold is chosen s.t. FAR=0.001, in my case i chose the threshold to be 0.2)


*data_augmentation_utils.py*      several helper functions for augmentating the data (flip, rotate, warp, add_noise etc.)

*models.py*                       implementation of the resnet with added layers (the actual one that i took in the end and finetuned it)

*train.py*                        trains the neural net, epoch, batch size, learning rate and many more can be set (make sure, that you have generated the XXX_DATA_augmented.npy and corresponding labels with the jupyter notebooks in 1_data_preprocessing). you have to run *generate_texas_data.py* inside 1_data_preprocessing first, so that the train data can be loaded. unfortunately you can only download the *TEXAS_AUGMENTED_faces.npy* since BDR data is private

*utils.py*                        many helper functions for: computing the loss, randomly take training data with a batch generator, load images and any more
