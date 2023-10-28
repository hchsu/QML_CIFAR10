# Transfer learning for image recognition of CIFAR-10
The pretrained CNN model is applied as a feature exptractor. 
Only the last block, the classification block, is trained with CIFAR-10. 
There are two kind of transfer learning:
1. classical-to-classical (c2c): The last block is a classical nerual network. 
2. classical-to-quantum (c2q): The last block contains at least a layer of quantum neurons. 

## Required Packages:
    Python 3.9 
    Pennylane 0.32.0 
    Torch 2.1.0
## The code development environment: 
    Mac OS 12.1 with intel CPU.

## To run the code: pass two arguments in the command line. 
 The first argument is a BOOL that decides whether to have quantum layer (1) or not (0).
 
The second argument is batch_size. For example:

    python transf.py 0 16

runs classical transfer learning with batch_size=16. 

For the two-class classification with batch size=16, classcial-to-classical transfer lerning takes 20 sec. per epoch.

The classical-to-quantum transfer learning takes 3 min. per epoch.

## The code is based on the reference:
    [1] Andrea Mari, Thomas R. Bromley, Josh Izaac, Maria Schuld, and Nathan Killoran. 
        Transfer learning in hybrid classical-quantum neural networks. Quantum 4, 340 (2020).
    [2] Reference code of [1]. https://github.com/XanaduAI/quantum-transfer-learning/tree/master
    [3] Pytorch tutorial. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
