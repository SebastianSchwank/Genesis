#Bi-Neuronal Network

- Error free, Derivative free learning
- Neuron locale solution
- The architecture is a multiple layered connected neural network with feedback connections
- It works without backpropagation
- It displays all layers in one matrix

Pro's:
- No backpropergation essential
- Multiple layers supported
- Matrix view of multilayer networks
- Good paralellisation thanks to self error calculating neurons 
- Debug window for the activity of the neurons over all lessons included
- Learning speed improvement VS. backpropagation in some cases

 In this demo the net learns to analyse the frequency for a given sine function (the count of the active neuron is the frequency).

 This is my "n"-th implementation of the so called "CONDITIONAL NEURAL NETWORK"
 It's inspired by the biological model of neurons so each neuron calculate it's error for itself
 and seems to be a way of training patterns to a random initalized
 neural net without the need of Backpropergation.
 I want to notice that the network developes from a stacked and completly connected graph
 exept there are no feedbacks in the neurons itselves.


The basic idea is that the weights change like this:
The activation of the net including input activation and target activation minus the net's activation only with input multiplied with the weight's neuron-input is the delta(W). Full Result propagated several times minus Propagation of the Input Stimulus only.

So you process the net only with input activation then you process the net with input and target activation. Then you subtract the two activation patterns of these two calculations and for each neuron you get a subtraction which you multiply with the neuron's input, the result you add to the current weight's value.

w[i][j] += activation_without_result[j]*(activation_with_result[i]-activation_without_result[i]);

As activation function I take the sigmoide function: https://de.wikipedia.org/wiki/Sigmoidfunktion

Screenshot of the weights matrix of the three layered network which learns to detect the frequency of a sampled sine wave:
![Screenshot](https://github.com/SebastianSchwank/ANN-Sofie/blob/master/ANN-Sofie-Generalisation.png)

If you want to give feedback to the calculus or the algorythm, don't hesitate to write an e-mail:
sebastian.schwank@googlemail.com

\sum \limits_{n=0}^{\infty}

License:

The permission is given only for ediucational.
This software is written by Sebastian Schwank. These three sentences have to be included in the Software License and it's permitted to remove them.

TODO:

Cuda Implementation
