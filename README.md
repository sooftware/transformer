[<img src="http://img.shields.io/badge/Documentation-passing-9cf">](https://sooftware.github.io/Attention-Is-All-You-Need/) <img src="http://img.shields.io/badge/License-Apache--2.0-9cf">
# Transformer
  
A PyTorch Implementation of Transformer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).  
This repository focused on implementing the contents of the paper as much as possible.  
  
## Intro 
  
<img src="https://tutorials.pytorch.kr/_images/transformer_architecture.jpg" height=500>  
  
This repository focused on implementing the contents of the paper as much as possible,   
while at the same time striving for a readable code. To improve readability,      
I designed the model structure to fit as much as possible to the blocks in the above Transformers figure.
  
## Get Started
  
The toy problem is brought from [IBM/pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq).  
  
### Prepare toy dataset  
```
$ generate_toy_data.sh --dir ../data --max_len 10
```  
  
### Train and play
```
$ toy.sh --d_model 512 --num_heads 8 --d_ff 2048
```
  
**TODO: be in the process of implementation !!**     
  
Once training is complete, you will be prompted to enter a new sequence to translate and the model will print out its prediction (use ctrl-C to terminate). Try the example below!  
  
```
Input: 1 3 5 7 9
Expected output: 9 7 5 3 1 <eos>
```
  
### Checkpoints  
Checkpoints are organized by experiments and timestamps as shown in the following file structure  
```
experiment_dir
+-- input_vocab
+-- output_vocab
+-- checkpoints
|  +-- YYYY_mm_dd_HH_MM_SS
   |  +-- decoder
   |  +-- encoder
   |  +-- model_checkpoint
```  
The sample script by default saves checkpoints in the `experiment` folder of the root directory. Look at the usages of the sample code for more options, including resuming and loading from checkpoints.
  
## Troubleshoots and Contributing
  
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/Awesome-transformer/issues) on Github.   
Or Contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Dependencies
  
* Python 3.6+
* Numpy
* Pytorch

## TODO  
  
* [ ] Implements Evaluator & Predictor
* [ ] Extending to speech recognition version 
  
### Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com

.
