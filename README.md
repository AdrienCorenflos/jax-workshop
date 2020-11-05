

# JAX-workshop
11 November 2020 - Aalto University

## Introduction

Over the past decade computational and statistical machine learning (ML) witnessed the advent of 
gradient-based learning and inference methods. Producing stable derivatives once was an expensive procedure, 
either requiring human power to derive practical formulas, or compute power to apply the chain rule symbolically. 
Where these two operations required to express the gradient of the functions at stake at any possible point, 
automatic differentiation took a radically different approach by enabling to compute of the derivatives only 
for these input/output values that the program needed to evaluate. This was done by coupling the composition graph 
of the program with that of the chain rule and gave birth to two main differentiation methods: the forward-mode gradient 
accumulation and the backward mode. 

Numerical differentiation libraries have since been built on this foundation, from the now defunct 
[HIPS Autograd](https://github.com/HIPS/autograd) and [Theano](https://github.com/Theano/Theano) to the widely used 
Facebook backed [PyTorch](https://pytorch.org/) and Google championed [TensorFlow](https://www.tensorflow.org/). 
Whilst the first-comers in the automatic differentiation race were only interested in the very fact of computing 
derivatives, due to the growing use of big (deep) models, the emphasis has recently been put on compiler and 
hardware specificities especially in the context of linear algebra operations. Two main advances highlight these: the 
first one is the creation of 
[tensor processing units](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu), 
a coprocessor specialized at large batch, low precision computations, and the second is the release of [accelerated 
linear algebra](https://www.tensorflow.org/xla/) (XLA), a static-graph optimisation/combination of linear algebra chains.

Additionally to this, a vast and profuse ecosystem of research-oriented and industrial companion packages
have developed around these two libraries, in particular TensorFlow, making the entry cost to gradient-based machine
learning steep for many people. [JAX](https://github.com/google/jax) is an alternative to these that stripped the 
accelerated automatic differentiation back to its basics whilst also enforcing from the very start a functional API 
where TensorFlow and PyTorch had favoured object-oriented interfaces and magical side-effects (leading to a lot of 
mystical effects) instead.

![A subjective comparison](./incl/tf-pytorch-jax.pdf)

## Workshop organisation

As a consequence, this workshop will be split into independent streams that will be subdivided into three 
different levels from beginner to advanced, which we will try and combine together at the end of the session.

The three different streams will be the following: randomness, controlflow and differentiability.

### Randomness

### Parallelization/Vectorization

### Controlflow

### Differentiability

