# CS231N Lecture 3

---

## Resources

Slides: https://cs231n.stanford.edu/slides/2025/lecture_3.pdf
 
Direct Video link: 
https://www.youtube.com/watch?v=dyNGd06MWn4&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=3

[![CS231N Lecture 3](https://img.youtube.com/vi/dyNGd06MWn4/0.jpg)](https://www.youtube.com/watch?v=dyNGd06MWn4)

---

## TLDR

- Regularization
- Stochastic Gradient Descent
- Momentum, AdaGrad, Adam
- Learning rate schedules

---

## Summary

### Regularization

$$L(W) = \frac{1}{N} \sum_{i=1}^N L_i + \lambda R(W)$$

$$L_i = Loss(f(x_i,W),y_i)$$

11:30 : Loss function and Regularization
- Loss function tells us how good, or how bad our current classifier is.
  - In optimization speak, this is an objective function that we want to minimize.
- Regularization prevents the model from doing *too* well on training data, i.e. do worse on training data to do better over test data
  - There are many ways to cause "Regularization", one of these is to introduce a secondary objective function during optimization.
  - Since we now have two optimization targets, first an objective or loss function, and second a regularization term, we tradeoff between the two optimization targets by using terms like $\lambda_1$ and $\lambda_2$. Numerically having two values is not important here, we just need a ratio between the two, so we use a single $\lambda$, however if we have 3 or more objectives to optimizes, we might consider $(1,\lambda_1,\lambda_2,\dots)$ appropriately.
- Specific examples of Regularization terms:
  - L2: $R(W) = \sum_k \sum_l W^2_{k,l}$
  - L1: $R(W) = \sum_k \sum_l | W_{k,l} |$
  - Elastic Net (L1+L2): $R(W) = \sum_k \sum_l \beta W^2_{k,l} + \sum_k \sum_l | W_{k,l} |$
- Other forms of introducing "Regularization" outcomes, where we prevent our model from doing *too* well on training data, while encouraging better generalizability to unseen test data, include the following:
  - Dropout
  - Batch Normalization
  - Stocahstic depth, fractional pooling, etc
- Why Regularize?
  - Express preferences over weights
    - eg: Want sparse weights for better interpretability? use L1 regularization
    - eg: Want smooth weights? use L2 regularization
  - Make the model simple so it generalizes better on test data
  - Improve optimization by adding curvature
    - eg: L2 optimization adds curvature, makes it faster to converge in some cases

---

### Optimization

24:20 : Optimization
- Optimization is how we will find the optimal $W$ or weights for our model
- Naive approach: *Random search* - i.e. sample 1000 points and just try them all
  - This works some of the time with low accuracy. But we can do better.
- Intuition
  - Navigating a *Loss landscape*, aka Loss manifold, can be thought of as a marble of a stream of water, rolling or flowing down a hilly terrain, letting gravity pull it down any slopes encountered to a valley or trough.
- How do we follow the slope?
  - We use the derivative. In 1-dimension:
$$\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
  - We could calculate the derivative with a numerical or analytical approach
  - Numerical approach: Perturb each component of $W$ and use the limit definition, loop through each value. This idea can be used for verification, but is not preferred in general since its slow, approximate and prone to errors (numerical precision errors) due to limited floating point arithmetic precision.
  - Analytical approach: We know our objective function mathematically, and using the chain rule, we can calculate the gradient analytically. This method is exact, fast, but is error-prone (implementation errors) - so its worth doing a final check with the numerical approach.
    - Most of the time, we know our loss function, and its usually convex and differentiable, which is why the analytical approach is preferred. One could possibly have a use-case involving a non convex or non differentiable loss function, in which case the analytical approach might not work. Most of the time this can be avoided.

#### Gradient Descent (GD)

- This is a standard optimization technique for convex functions

```python
# Vanilla Gradient Descent

while True:
  weights_grad = evaluate_gradient(loss_function, data, weights)
  weights += - step_size * weights_Grad # perform parameter update
```

Challenges of Vanilla GD
- Calculating the full sum is expensive when $N$ is large

  $$L(W) = \frac{1}{N} \sum_{i=1}^N L_i (x_i, y_i, W) + \lambda R(W)$$
  
  $$\nabla_W L(W) = \frac{1}{N} \sum_{i=1}^N \nabla_W L_i (x_i, y_i, W) + \lambda \nabla_W R(W)$$

#### Stocastic Gradient Descent (SGD)

- We address the expensive full sum calculation in GD, by doing small batches, or mini-batches, say size 32, 64 or 128. This is technically called Mini-batch Gradient Descent.

- In some literature Stochastic Gradient Descent refers to the idea of taking gradient steps after seeing a batch-size=1 records of data, while other literature appears to define stochastic gradient descent the same as minibatch gradient descent.

- In either case, once we've looped through all the data once, we'll call it the completion of one epoch, and loop through the data again.

$$\text{SGD}:$$
$$x_{t+1} = x_t - \alpha \nabla f(x_t)$$

```python
# Vanilla Minibatch Gradient Descent

while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_function, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```

Problems with SGD:
1. Loss changes quickly in one direction and slowly in another
   - could overshoot
   - could jitter around, i.e. make slow progress along shallow dimension and jitter along steep direction
   - this happens when the Loss function has a **high condition number**
     - **condition number**: the ratio of largest to smallest singular value of Hessian matrix
     - one interpretation is that loss changes quickly in one direction, and slowly in another
2. Local minima or Saddle point
   - Zero gradient might get us stuck
   - eg: visualize $x^2-y^2$
   - Note: Saddle points are common as we get into higher dimensions
3. Noisy updates from subset of data
   - Since we are examining a subset of our data at each step, we have some inherent noise at each step

#### SGD + Momentum

[42:56](https://youtu.be/dyNGd06MWn4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&t=2577) 

- [2013, Sutskever et al](https://proceedings.mlr.press/v28/sutskever13.html)
- One trick to address all of the issues in SGD, is to introduce momentum
- Intuition - this is very much like a ball rolling down a hill.
- This addresses all the issues in SGD
  - Momentum helps breaks through local minima and saddle points
  - Momentum smooths issues with minibatch noise
  - Momentum helps address poor conditioning (loss changing quickly in one direction, and slowly in another) by building more momentum along the steeper direction
- As the slides note: 
  - Build up "velocity" as a running mean of gradients
  - $\rho$ gives "momentum"; typically $\rho=0.9$ or $0.99$

$$\text{SGD} +\text{Momentum}:$$ 
$$v_{t+1} = \rho v_t + \nabla f(x_t)$$ 
$$x_{t+1} = x_t - \alpha v_{t+1}$$

```python
# SGD Momentum

vx = 0
while True:
  dx = compute_gradient(x)
  vx = rho * vx + dx
  x -= learning_rate * vx
```

Alternate but equivalent formulation:

$$\text{SGD} +\text{Momentum (alternate)}:$$
$$v_{t+1} = \rho v_t - \alpha \nabla f(x_t)$$
$$x_{t+1} = x_t + v_{t+1}$$

```python
# SGD Momentum (alternate) 

vx = 0
while True:
  dx = compute_gradient(x)
  vx = rho * vx - learning_rate * dx
  x += vx
```

Challenges of SGD + Momentum
- Overshooting - we risk a big overshoot when we reach our optimal convergence, because momentum keeps us going, but eventually brings us back to the optimal

#### RMSProp

[49:25](https://youtu.be/dyNGd06MWn4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&t=2962)

- RMSProp came out in 2012, from Geoffrey Hinton's group
  - Geoff Hinton, "Neural Networks for Machine Learning" (lecture 6a notes: RMSProp). 2012. PDF: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- Intuition: 
  - Lets take larger steps in flatter parts of Loss landscape, but smaller steps in steeper parts of the landscape
- Mathematically we do this by introducing adaptive scaling of the gradient components 
  - i.e. scale more in some cases, less in others


```python
# RMSProp

grad_squared = 0
while True:
  dx = compute_gradient(x)
  grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
  x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

- Breakdown
  - Keep a running average of the grad\_squared
    - decay\_rate is a much like the momentum term, but now its on the squared gradient
    - Past terms are scaled with decay\_rate, new values with (1-decay\_rate)
  - Divide learning\_rate by cumulative grad\_squared
    - i.e. Where the gradient is larger (irrespective of direction), we'll make smaller steps - but when the gradient is small, we'll make larger steps
    - This changes the way we step in steep vs shallow directions

Quiz
- How does our gradient step direction change in RMSProp in the division by squared gradient values?
  - When we have large values, we divide by even larger values, so we reduce the effective step size
  - When we have small values, we divide by smaller values, so we take larger effective steps
  - This is potentially what makes RMSProp better than SGD+Momentum, since it prevents overshooting

#### Adam

[53:32](https://youtu.be/dyNGd06MWn4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&t=3212)


Adam (almost)
```python
# Adam (almost)

first_moment = 0
second_moment = 0
while True:
  dx = compute_gradient(x)
  first_moment = beta1 * first_moment + (1 - beta1) * dx               # Momentum
  second_moment = beta2 * second_moment + (1 - beta2) * dx * dx        # RMS Prop
  x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)  # RMS Prop
```

Quiz
- The above "Adam (almost)" implementation fails in the first timestep. Why?
  - Hint: The problem is in the second moment calculation
  - Note: `beta1, beta2` are initialized close to 1, first and second moments are initialized to 0
  - When we update $x$, the denominator is 0 in the first timestep. This creates a very large initial step even if the gradient is very small. 
- Adam adds bias terms to account for this. See full form below.
  - Bias correction addresses the fact that first and second moment estimates start at zero
  - Good starting point for many models, initialize Adam with `beta1=0.9`, `beta2=0.999`, `learning_rate=1e-3` or `5e-4`
  - Adam now has characteristics of both RMSProp and SGD+Momentum

Adam (full form)
```python
# Adam (full form)

first_moment = 0
second_moment = 0
for t in range(1, num_iterations):
  dx = compute_gradient(x)
  first_moment = beta1 * first_moment + (1 - beta1) * dx # Momentum    # Momentum
  second_moment = beta2 * second_moment + (1 - beta2) * dx * dx        # AdaGrad / RMSProp
  first_unbias = first_moment / (1 - beta1**t)                         # Bias correction
  second_unbias = second_moment / (1 - beta2**t)                       # Bias correction
  x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)  # AdaGrad / RMSProp

```

#### AdamW: Adam Variant with Weight Decay

[57:19](https://youtu.be/dyNGd06MWn4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&t=3438)

TLDR: AdamW separates the weight decay step from the gradient updates

Quiz
- How does Regularization interact with the Optimizer? (eg: L2)
  - A: It depends - basically gets caught up in the first and second moment calculations

- Adam vs AdamW - difference lies in how they treat regularization
  - Adam - couples weight decay with adaptive moments
    - ie the `dx=compute_gradient(x)` line involves the loss and weight decay term
    - so we calculate first and second moments on the weight decay also
  - AdamW - decouples weight decay and adds it at the end
    - ie the `dx=compute_gradient(x)` line is about the loss landscape only
    - we add the weight decay at the end, after the last step, `x -= \dots` 
  - Adam's adaptive scaling affects the L2 penalty also, scaling it with per-parameter learning rates in messy unintended ways
  - Effectively Adam is butchering the effect of the regularization term, and AdamW is restoring the correct behavior of the regularization term, in ways comparable to SGD

#### Optimization Summary

- **GD**: Each step is tedious, have to use the entire dataset each time.
- **true SGD** is stochastic, one sample at a time. Very fast.
- **Mini-batch GD**: (referred to as **SGD** in cs231n notes): Quite effective, but has potential issues with saddle points, poor conditioning, and noise related from mini-batch sampling
- **SGD+Momentum**: introduces Momentum (first moment) to mitigate the issues with saddle points, conditioning and batch related noise
- **RMSProp**: introduces second moment, a different scaling of the gradient for each parameter, effective getting **per-parameter learning rates**, or **adaptive learning rates**
- **Adam**: combines ideas from SGD+Momentum and RMSProp, by using both first and second moments
- **AdamW**: improves upon Adam by fixing the Regularization issue where Adam does first and second moments on both the loss and regularization terms. RMSProp also has this problem but is a much smaller issue in RMSProp than in Adam.


---

### Learning Rates

[59:15](https://youtu.be/dyNGd06MWn4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&t=3555)

Learning Rates as a Hyperparameter

- Fixed vs Varying/Decaying Learning Rate?
  - All modern deep learning implementations use a varying learning rate that decays over time
    1. STEP
        - Reduce learning rate at fixed points, say `LR*=0.1` at epochs $30, 60, 90, \dots$
        - Common in ResNets
    2. COSINE: $\alpha_t = \frac{1}{2} \alpha_0 (1 + \cos (t \pi / T))$
       - $\alpha_0$: Initial Learning Rate
       - $\alpha_t$: Learning Rate at epoch $t$
       - $T$: Total number of epochs
    3. LINEAR: $\alpha_t = \alpha_0 (1 - t / T)$
    4. INVERSE SQRT: $\alpha_t = \alpha_0 / \sqrt{t}$
- LINEAR WARMUP
  - Increase LR initially during say first ~5000 iterations
    - High Initial learning rates can make loss explode
    - linearly increasing learning rate from 0 over the first ~5000 iterations can prevent this.
- **Empirical rule of thumb (aka Linear Scaling Law)**: 
  - If you increase the batch size by $N$, also scale the initial learning rate by $N$

---

### Second Order Optimization

- Second Order Optimization is not a major topic in CS231N
- The idea is to use the Hessian, ie fit a quadratic approximation to find a minima with a Newton parameter update
  - Requires a Taylor expansion, needs a second derivative, which may be difficult
  - Derivative may get very large

$$J(\theta) \approx J(\theta_0) + (\theta - \theta_0)^T \nabla_\theta J(\theta_0) + \frac{1}{2} (\theta - \theta_0)^T H(\theta_0)$$
$$\theta^* = \theta_0 - H^{-1} \nabla_\theta J(\theta_0)$$

- Why is this bad for Deep Learning?
  - Hessian has $O(N^2)$ elements
  - Inverting takes $O(N^3)$ 
  - $N =$ (tens or hundreds of) millions
- Works very well for small problems when training small models or have a lot of time and computational power, but for larger problems in deep learning or more generally, we learn more from seeing more data.

---

### LECTURE 3 TAKEWAYS

[1:05:58](https://youtu.be/dyNGd06MWn4?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&t=3956)

- **Adam(W)** is a good default choice in many cases. Works ok even with constant learning rate.
- **SGD + Momentum** can outperform Adam, but may need more LR tuning and schedule
- If you can afford full batch updates, then look beyond 1st order optimization (**2nd order and beyond**)

