# NN_From_Scratch
### Follow-on notebooks used while studying the basics of Neural Networks and Language Models
- Micrograd From Scratch: Defining the auto-diff process using handmade classes, in a similar way as PyTorch
- Makemore: Bigram char-level model for creating new names, using a list of 32033 names as dataset. Study of concepts such as logits, softmax, Negative Log Likelihood as Loss function and Gradient descent. Neural Network built from scratch using torch tensors.
- Part 2: Makemore, now using embeddings, learning rate decay and minibatches. Also used batch normalization and verify how the paramaters distribution flow through the NN.
- Part 3: Study the backpropagation process, calculating each gradient by hand, since the loss function until the embeddings and checking if it matches PyTorch gradients. No features were added to Makemore.
- Part 4: Makemore, now using a more complex model (WaveNet architecture). Also the different layers were modularized in classes. Use of flatten layers