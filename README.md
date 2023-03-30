_____________________________________________________________________________________________________________________________

This Repository contains the implementation belonging to the paper "Protecting 3D Face Embeddings".
_____________________________________________________________________________________________________________________________


## 1_data_preprocessing
contains the code for the preprocessing of the rendered data (cropping, resizing, scaling to [-127,127])

## 2_neural_net_training
contains the code for model training and prediction

## 3_SortedPolyProtect
contains the code for the whole analysis of the Biometric Template Protection Method "SortedRadicals"

## DATA_depth_images_raw
put inside here (inside DATA_depth_images_raw/opensource_DATA/texas/ ) the face images of the Texas database (**at least the folder /PreprocessedImages  which you get from https://live.ece.utexas.edu/research/texas3dfr/)**,
those will be used by the jupyter notebooks inside 1_data_preprocessing

Inside each folder there is a separate readme.txt for further explanation.
_____________________________________________________________________________________________________________________________

# Requirements:
tested on
Ubuntu 22.10
Python 3.10.7
put the face images of the Texas database (at least the folder /PreprocessedImages  which you get from https://live.ece.utexas.edu/research/texas3dfr/) inside DATA_depth_images_raw/opensource_DATA/texas/

# Installation:
clone this repository, cd into the root directory of this repository,

it is recommended to create a virtual python environment in order to use this code e.g. via
```bash
python3 -m venv NAME_OF_YOUR_ENVIRONMENT
```

(if you are currently inside the root directory of this repo, otherwise add the absolute path to ../../NAME_OF_YOUR_ENVIRONMENT,
further explanation in https://docs.python.org/3/library/venv.html)

activate this environment (deactivation is done by simply executing "deactivate")
```bash
source NAME_OF_YOUR_ENVIRONMENT/bin/activate
```

install python packages:
```bash
pip3 install -r requirements.txt
```

add kernel to jupyter notebook
```bash
python3 -m ipykernel install --user --name=NAME_OF_YOUR_ENVIRONMENT
```


and you should be ready to go
(inside the jupyter notebook go onto kernel -- change kernel and select your python environment)
_____________________________________________________________________________________________________________________________

# BUGFIXES:

it is very likely that the error "No module named 'keras.engine.topology'" will appear, when executing some of the python scripts. bug fix:
https://stackoverflow.com/questions/68862735/keras-vggface-no-module-named-keras-engine-topology
(CHANGE from keras.engine.topology import get_source_inputs TO from keras.utils.layer_utils import get_source_inputs inside ~/.../CODE_3D_FACE/NAME_OF_YOUR_ENVIRONMENT/lib/python3.10/dist-packages/keras_vggface/models.py)


