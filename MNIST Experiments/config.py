batch_size  = 32
img_h       = 28
img_w       = 28
num_labels  = 10

# Hyperparameters

step_size = 0.001
total_epochs = 100
total_iterations = int(60000/batch_size)

# Network Parameters

# Layer 1 

conv_1 = {
            'filters'       : 6,
            'kernel_size'   : (3,3),
            'strides'       : (2,2)
         }

# Layer 2

conv_2 = {
            'filters'       : 5,
            'kernel_size'   : (3,3),
            'strides'       : (2,2)
         }

# Layer 2

conv_3 = {
            'filters'       : 4,
            'kernel_size'   : (2,2),
            'strides'       : (2,2)
         }


model0_params = {
                    'conv_1' : conv_1,
                    'conv_2' : conv_2,
                    'conv_3' : conv_3
                }

#------------------------------------------------------------------------------