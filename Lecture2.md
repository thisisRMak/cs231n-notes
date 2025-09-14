# CS231N Lecture 2

---

## Resources

Slides: https://cs231n.stanford.edu/slides/2025/lecture_2.pdf
 
Direct Video link: 
https://www.youtube.com/watch?v=pdqofxJeBN8&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=2

[![CS231N Lecture 2](https://img.youtube.com/vi/pdqofxJeBN8/0.jpg)](https://www.youtube.com/watch?v=pdqofxJeBN8)

---

## TLDR

Two types of Image classifiers

1. Nearest Neighbors: kNN
2. Linear Classifiers: softmax, SVM

---

## Summary

### Image Classification - The Problem

00:00 : Image Classification - What is it? 

05:00 : Image Classification - Why is it hard? Semantic Gap? 
- Illumination / Shadows 
  - eg: cats in shadows
- Background Clutter
  - eg: white cat in a field of snow, brown cat in a dirt hill
- Occlusion 
  - eg: cat hiding under a cushion, only the tail showing
- Scale / Zooming in and out
- Resolution 
  - Note: not considered as big an issue, since we normalize the image
- Deformation
  - eg: cats sitting or lying, looking deformed
- Interclass variation 
  - eg: different colored cats
- Context 
  - eg: dog with shadow vs tiger

11:52 : Interface for an Image Classifier
```python
def classify_image(image):
  # Some magic here?
  return class_label
```

12:15 : Why not hardcode? Why use ML?

- Hardcoding or ideas like edge-detection have limited results despite considerable effort
- ML approach uses large amount of data to "learn"
  1. Collect a dataset of images and labels
  2. Use ML algorithms to train a classifier
  3. Evaluate the classifier on new images

```python
def train(images, labels):
  # Machine Learning!
  return model
def predict(model, test_images):
  # Use model to predict labels
  return labels
```

### Image Classification using Nearest Neighbors

16:50 : Nearest Neighbor Classifier
- "Nearest" determined by Pixel-wise difference between images
  - Many methods exist for fast/approx nearest neighbor. A good implementation: https://github.com/facebookresearch/faiss
- Time analysis
  - Training: $O(1)$
    - Memorize everything
  - Prediction: $O(N)$
    - Scan and diff input image with the corpus
- HyperParameters
  - What is the best value of $K$ to use?
    - Pick $K$ nearest neighbors and do majority voting
    - We may end up with regions where we can't reach a decision. **Tip**: Collect more data for these regions.
  - What is the best distance to use?
    - L1 (Manhattan) distance: $d_1(I_1,I_2) = \sum_p | I_1^p - I_2^p|$
      - preserves features better, sensitive to rotation
    - L2 (Euclidean) distance: $d_1(I_1,I_2) = \sqrt { \sum_p (I_1^p - I_2^p)^2}$
  - Demo to try kNN: http://vision.stanford.edu/teaching/cs231n-demos/knn/

#### [Side Quest] Hyperparameters - Tuning and Cross Validation

35:25 : Hyperparameters
- Variables that a user/designer has to pick
- May be dataset dependent or problem dependent
- Selecting Hyperparameters is called Hyperparameter Tuning
- Idea #1: Choose Hyperparameters that work best on training data
  - Bad because we have 100% accuracy on training data and no measure for unseen data
- Idea #2: Choose hyperparameters that work best on test data
  - Bad because no idea how algorithm will perform on new, unseen data
- Idea #3: Split data into train, val; choose hyperparameters on val and evaluate on test
  - Better!

39:43 : Cross Validation

- Idea #4: Cross-Validation: Split data into folds, try each fold as validation and average the results
  - Useful for small datasets, not too frequently used in Deep Learning
  - Only run on the test set once at the very end

#### [Side Quest] CIFAR10 dataset

- Data
  - 10 classes
  - 50,000 training images
  - 10,000 testing images
- 5-fold Cross-validation shows $k~=7$ yields $29\%$ accuracy, much better than random guessing, which would yield $10\%$ accuracy

44:10 : kNN with Pixel Distance is rarely used
- Distance metrics on pixels are not informative
  - eg: Can't meaningfully differentiate Occlusion vs Pixel Shift vs. Tint

### Image Classification using Linear Classifiers

46:20 : Linear Classifier
- Parametric approach
$$\text{Image} \rightarrow f(x,W) \rightarrow \text{10 numbers giving class scores}$$
$$x = \text{flattened vector representation of image of shape }32 \times 32 \times 3 = 3072 \text{ values}$$
$$\underbrace{f(x,W)}_{10 \times 1} = \underbrace{W}_{10\times 3072} \underbrace{x}_{3072\times 1} + \underbrace{b}_{10\times 1}$$

53:28 : Challenges for a Linear Classifier
- XOR like classifications
- a donut around the origin
- multiple clusters of data

56:20 : How to think about a Loss function

- Given a dataset of examples $\{(x_i,y_i)\}^N_{i=1}$
  - $x_i :=$ image
  - $y_i :=$ integer label
- Loss over dataset is the average of loss over examples
  - $L=\frac{1}{N} \sum_i L_i (f(x_i,W),y_i)$

#### Softmax Classifier (Multinomial Logistic Regression)

57:28 : Softmax Classifier

- Interpret raw classifier scores as probabilities
- Optimize by solving a Maximum Likelihood Estimation

$$s = f(x_i; W)$$
$$P(Y=k|X=x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}}$$
$$L_i = - \log P(Y = y_i | X = x_i)$$

- Other approaches 
  - minimize KL divergence, i.e. Kullback-Leibler divergence
$$D_{KL}(P||Q) = \sum_y P(y) \log \frac{P(y)}{Q(y)}$$
  - minimize Cross Entropy Loss
$$H(P,Q)=H(p) + D_{KL}(P||Q)$$

- Min/Max possible Softmax loss $L_i$? 
  - $p: 0 \to 1$, $L_i = - \log p = \infty \to 0$
- At initialization, all $s_j$ will be equal, what is $L_i$ assuming $C$ classes?
  - $L_i = -\log(1/C) = \log C = \log 10 \approx 2.3$

#### SVM Classifier

(not covered in video, but included in slides)

$$\begin{align*}
L_i 
&= \sum_{j \neq y_i} \begin{cases}0 & \text{if } s_{y_i} \geq s_j + 1 \\ s_j - s_{y_i} + 1 & \text{otherwise}\end{cases}
\\
&= \sum_{j \neq y_i} \max \left( 0, s_j - s_{y_i} + 1\right)
\end{align*}
$$