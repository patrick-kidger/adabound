# AdaBound for Keras

Keras port of [AdaBound Optimizer for PyTorch](https://github.com/Luolc/AdaBound), from the paper [Adaptive Gradient Methods with Dynamic Bound of Learning Rate.](https://openreview.net/forum?id=Bkg3g2R9FX)

Use as a dropin replacement for `Adam` Optimizer. 

```python
from adabound import AdaBound

optm = AdaBound(lr=1e-03,
                final_lr=0.1,
                gamma=1e-03,
                weight_decay=0.,
                amsbound=False)
```

# Requirements
- Keras 2.2.4+ & Tensorflow 1.12+ (Only supports TF backend for now).
- Numpy
