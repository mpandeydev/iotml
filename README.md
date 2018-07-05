# vanilla_nn.py Usage

* To be used with **training_data.csv** only.
* Place dataset file one directory above code file (For Git repo purposes)

## Running the program
1. Set up a virtual environment to include `tensorflow`,`matplotlib`,`numpy` and `csv` ( I have used `anaconda` to set it up, so the commands following require that you have anaconda installed)
2. In command line, run your environment by using `activate your_environment_name`
3. Navigate to the location of `vanilla_nn.py`
4. Run using `pyhon vanilla_nn.py`

vanila_nn.py is a Deep neural Network implementation which is used to classify magnetometer readings from vehicles to identify their speeds.

## Specifications

* Feature Data shape = [1,4800] X,Y,Z axis magnetometer readings low passed and truncated 
* Label Data shape = [1] Speed of vehicle
* Dataset size = 9006 (normalized by `tf.keras.utils.normalize`
* Dataset split : 80:10:10 Training, Validation, Testing

## Architecture

* Deep Neural Network with 7 layers(~)
* ReLU activations with a final layer of Softmax
* Loss : Sparse softmax cross entropy for exclusive classes `tf.losses.sparse_softmax_cross_entropy`
* Optimizer : `tf.train.AdagradOptimizer`
* Epochs : 100,000

Number of epochs, Nodes in hidden layers and batch size to be varied

## To be done

* Batch Normalization
