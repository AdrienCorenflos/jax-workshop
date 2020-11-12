

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
learning steep for many people. 

[JAX](https://github.com/google/jax) is an alternative to these ecosystems that stripped the 
accelerated automatic differentiation back to its basics whilst also enforcing from the very start a functional API 
where TensorFlow and PyTorch had favoured object-oriented interfaces and magical side-effects (leading to a lot of 
mystical effects) instead. In particular, through just-in-time compilation (JIT) to XLA, it is focused on delivering a high performance numerical computing and automatic 
differentiation framework with an emphasis on seamless batching and replicable parallelisation across devices. 

![JAX logo](https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png)

## Workshop organisation

As a consequence, this workshop will be split into independent streams that will be subdivided into three 
different levels from beginner to advanced, which we will try and combine together at the end of the session.

The three different streams will be the following: randomness, controlflow and differentiability.

### Controlflow
At the core, JAX is fundamentally an API for C++, in particular for the XLA compiler. The controlflow notebooks will
teach you why it's important to use JAX primitives to communicate efficiently with the compiler and how to do so. 

### Parallelisation/Vectorisation
One of the USPs of JAX is its promise of a world free of tedious shape broadcasting through the use
of batching syntactic sugar. We will show how to implement these, how this translates in the compiled code, and we will
finally have a look at multiple devices parallelisation (this will not work online but instead I will run it on my machine).

### Randomness
Because JAX targets replicable parallelisation, as a consequence, it doesn't use global seeding like numpy or matlab 
(and other languages or libraries do), but instead relies on the 
[Threefry](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf) counter random number generator to enforce
reproducibility in varying hardware environments. We will see what happens when we use global seeds and multiprocessing, 
and how JAX implements the solution to this problem. We will also take a look at a few gotchas that one needs to pay 
attention to.

### Differentiability
And of course finally, JAX wouldn't be an automatic differentiation (AD) library if there was no differentiation involved. 
The differentiability notebooks will present the two main implementations of AD (forward-mode and reverse-mode), 
how to use gradients in JAX and how to implement your own custom gradients (and when it's needed). 


## Notes
Whilst some effort has been put into making all this material somewhat self-contained, some basic familiarity with 
[NumPy](https://numpy.org/) and to a lesser extent [SciPy](https://www.scipy.org/) is recommended if not expected.
