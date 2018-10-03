# 1D CNN for Vehicle Speed and Type Detection

## File Structure

- ```.. / datasets``` : All datasets should be contained in a folder named **datasets** one level above Git repo.

- ```1d_convolutional_network``` : Contains TensorFlow code for generating and training 1D Convolutional graphs, as well as scripts for inferencing and preforming additional post training operations on the model.
  - ```/3ch_cnn_quantized_updated.py``` : Script to generate 1D CNN graph and train it using training dataset _TODO : Clean code and keep vanilla implementation of 1D CNN._
  - ```/inference_non_shared.py``` : Script to use trained model to inference using subsets of dataset
  - ```/pruning.py``` : _TODO : Script for pruning individual neurons from each kernel_
  
- ```trained_models``` : Contains trained TensorFlow models as Checkpoint and Meta files

- ```trained_weights``` : Contains Numpy arrays of saved weights. These are only meant for analytical purposes, and not to be used in actual inferencing algorithms. To do so, restore weights from trained models.

## Usage

- Download datasets from [here](https://drive.google.com/drive/folders/113brHUKjoL7G4Ylv8XASAufMxLbHNbz5?usp=sharing) to a folder ```datasets``` as mentioned in the file structure.

- ### Training
  - Open ```1d_convolutional_network/3ch_cnn_quantized_updated.py```
  - Change hyperparameters as needed and remove call to ```save_network``` and ```tf.saver.save``` if you don't want to save your weights
  - Run file
  
- ### Inferencing
  - Open ```1d_convolutional_network/inference_non_shared.py```
  - Change **ONLY** any of : ```batch_size``` or ```step_size```
  
## Task List

[ ] Clean up code to remove redundant or unnecessary functions
[ ] Pruning algorithm for 1D Convolution Layers
