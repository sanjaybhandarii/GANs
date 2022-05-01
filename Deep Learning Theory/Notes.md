1.  Batch Normalization
    
    ->Normalization is done for a single neuron/perceptron for a batch of inputs i.e contains inputs from identical channels of different examples.
        It means that all channels 0 form a batch, channels 1 another batch and so on.

    i.Problems with Batch Normalization

        -> Normalization cannot be done until the processing layer(convolution) hasnot processed all the inputs in the batch. So , this creates problem in RNN.
        -> Performance depends on batch size i.e higher batch size means greater Performance.

2. Layer Normalization

    -> Normalization is done for a single example of input for a layer of network i.e contains inputs from all channels in a example.


3. Understanding GAN Loss 
    
    ![alt text](http://i.stack.imgur.com/2WU5Y.png)