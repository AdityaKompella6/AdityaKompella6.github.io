## Accelerating PyTorch: Custom CUDA Extensions for Unparalleled Performance

### Introduction
Although creating custom CUDA extensions to try to improve the performance of PyTorch may sound daunting,
it is simpler than you may think. I have a lot of experience in PyTorch coming from an ML
background but only have recently picked up on CUDA programming from taking a Parallel Programming
Course. I hope this blog post shows how even a relative beginner at CUDA can come up with a specific application,
write a CUDA kernel to speed it up, wrap that CUDA kernel so that it can be executed in Python and use it
in downstream tasks.
### Algorithm
The algorithm I chose to focus on this
### Pytorch Implementation

### How to Develop CUDA custom C++ extensions for PyTorch

### CUDA Implementation

### Benchmarks

### Conclusion

#### Some Python Code
```python
import torch
a = torch.pow(3,2)
print(a)
```
