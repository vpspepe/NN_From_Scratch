# NN_From_Scratch
### Follow-on notebooks used while studying the basics of Neural Networks and Language Models
- Micrograd From Scratch: Defining the auto-diff process using handmade classes, in a similar way as PyTorch
- Makemore: Bigram char-level model for creating new names, using a list of 32033 names as dataset. Study of concepts such as logits, softmax, Negative Log Likelihood as Loss function and Gradient descent. Neural Network built from scratch using torch tensors.
- Part 2: Makemore, now using embeddings, learning rate decay and minibatches. Also used batch normalization and verify how the paramaters distribution flow through the NN.
- Part 3: Study the backpropagation process, calculating each gradient by hand, since the loss function until the embeddings and checking if it matches PyTorch gradients. No features were added to Makemore.
- Part 4: Makemore, now using a more complex model (WaveNet architecture). Also the different layers were modularized in classes. Use of flatten layers
- NanoGPT Notebook: Used only to study the self-attention mechanism, such as Key, Query and Value concepts. Not pretty functional
- NanoGPT script: Full implementation of the Char-Level GPT, with the transformers architecture. It was also used skip connections, dropout layers and layer-normalization. Usefull classes/functions are inside the `gpt` folder. It can reuse pre-trained models to sample and generate some text (it was originally trained on Shakespeare books) or to train new ones. The hyperparameters of the new model should be setted in a `.json` file, and the input text in a `.txt` file (look at `bigram_params.json` and `input.txt`, as they are the default files used when the script is runned). You can also use different files/models by passing CLI args:
- - `--model_file`: must be a `.pth` file with your pre-trained model. If passed, will only generate some new text for fun :)
- - `--text_file`: must be a `.txt` file with the text you want to train (useless if `--model_file` is given). If not given, default to `input.txt`
-- `hyperparams_file`: must be a `.json` with the exactly same template as `bigrams.json` (useless if `--model_file` is given). If not given, default to `bigram_params.json`
