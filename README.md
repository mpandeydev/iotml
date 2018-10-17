# 1D CNN for Vehicle Speed and Type Detection

## File Structure

- ```.. / datasets``` : All datasets should be contained in a folder named **datasets** one level above Git repo.

- ```1d_convolutional_network``` : Contains TensorFlow code for generating and training 1D Convolutional graphs, as well as scripts for inferencing and preforming additional post training operations on the model.
  - ```/1d_cnn_training.py``` : Script to generate 1D CNN graph and train it using training dataset _TODO : Clean code and keep vanilla implementation of 1D CNN._
  - ```/1d_cnn_inference.py``` : Script to use trained model to inference using subsets of dataset
  - ```/pruning.py``` : Script to mask insignificant kernels using an L1 Norm importance metric as described in [Li et al. (2017)q](https://arxiv.org/pdf/1608.08710.pdf)
  
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
  - Select dataset on which to inference, selectable modes are ```speeds``` and ```types```
  - Change **ONLY** any of : ```batch_size``` or ```step_size```

- ### Pruning
  - Open ```1d_convolutional_network/pruning.py```
  - A full precision model is loaded by default, if required, change checkpoint and modify architecture accordingly
  - Function ```inference()``` runs a forward pass of network for given number of ```generations```
  - use function ```prune_layer()``` to prune a given layer
    - Inputs : 
      - ```tensor``` : A ```tf.Tensor``` of trained weights from layer to be pruned
      - ```next_tensor``` : A ```tf.Tensor``` of trained weights from next layer (For final layer, use ```None```)
      - ```num_prune``` : Number of kernels to be pruned from current layer
  
## Task List

- [X] Clean up code to remove redundant or unnecessary functions
- [X] Pruning algorithm for 1D Convolution Layers
