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

$$L(W) = \frac{1}{N} \sum_{i=1}^N L_i(f(x_i,W),y_i) + \lambda R(W)$$


11:30 : Loss function and Regularization
- Loss function tells us how good, or how bad our current classifier is
- Regularization prevents the model from doing *too* well on training data, i.e. do worse on training data to do better over test data
- Types of Regularizers
  - L2: $R(W) = \sum_k \sum_l W^2_{k,l}$
  - L1: $R(W) = \sum_k \sum_l | W_{k,l} |$
  - Elastic Net (L1+L2): $R(W) = \sum_k \sum_l \beta W^2_{k,l} + \sum_k \sum_l | W_{k,l} |$
