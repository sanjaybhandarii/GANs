1.  Batch Normalization
    
    ->Normalization is done for a single neuron/perceptron for a batch of inputs i.e contains inputs from identical channels of different examples.
        It means that all channels 0 form a batch, channels 1 another batch and so on.

    i.Problems with Batch Normalization

        -> Normalization cannot be done until the processing layer(convolution) hasnot processed all the inputs in the batch. So , this creates problem in RNN.
        -> Performance depends on batch size i.e higher batch size means greater Performance.

2. Layer Normalization

    -> Normalization is done for a single example of input for a layer of network i.e contains inputs from all channels in a example.
    -> Here, you dont need to wait for all the inputs in the batch to be processed so it is faster method.


3. Instance Normalization

    ->Normalization is done for a single channel by computing mean and std deviation for single individual channel.


4. Group Normalization

    -> Normallization is done by grouping a number of channels i.e compute mean and std dev based on a group(2,3 or more channels in a group) and then normalize.


5. Comparison of Normalization Techniques
![Image](https://user-images.githubusercontent.com/53966090/166148592-10cdbf9b-4ee8-4476-8bdd-505f478e3de6.png)
    Credit : https://www.youtube.com/watch?v=ffQfk3uS9Os

    -> As we can see that Instance Normalization doesnot perform well in Resnet-50 but Batch normalization and Group Normalization performs better.

    

5. Understanding GAN Loss 

    ->For GANs the losses are very non-intuitive. Mostly it happens down to the fact that generator and discriminator are competing against each other, hence improvement on the one means the higher loss on the other, until this other learns better on the received loss, which screws up its competitor, etc.

    Now one thing that should happen often enough (depending on your data and initialisation) is that both discriminator and generator losses are converging to some permanent numbers, like this:
    
    ![Image](http://i.stack.imgur.com/2WU5Y.png)

    This loss convergence would normally signify that the GAN model found some optimum, where it can't improve more, which also should mean that it has learned well enough.

    Here are a few side notes, that I hope would be of help:

    -If loss haven't converged very well, it doesn't necessarily mean that the model hasn't learned anything - check the generated examples, sometimes they come out good enough. Alternatively, can try changing learning rate and other parameters.

    -If the model converged well, still check the generated examples - sometimes the generator finds one/few examples that discriminator can't distinguish from the genuine data. The trouble is it always gives out these few, not creating anything new, this is called mode collapse. Usually introducing some diversity to your data helps.

    -This is some common sense but still: like with most neural net structures tweaking the model, i.e. changing its parameters or/and architecture to fit your certain needs/data can improve the model or screw it.

        Credit : https://stackoverflow.com/questions/42690721/how-to-interpret-the-discriminators-loss-and-the-generators-loss-in-generative
