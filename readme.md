## Problem 1

Given that the output of neuron in a multilayer perception is:
$$
x_{kj} = f\left(\sum^{N_{k-1}}_{i=1}(u_{kji}x^2_{k-1,i}+v_{kji}x_{k-1,i})+b_{kj}\right)
$$

where f is the sigmoid activation, given by the equation
$$
f(x) = \frac{1}{1+\exp(-x)}\label{sigmoid}\\
$$
and $v$ is used to define the matrix operation as follows:
$$
m_j =\sum^{N_{k-1}}_{i=1}(u_{kji}x^2_{k-1,i}+v_{kji}x_{k-1,i})+b_{kj}\label{v_matrix_op}\\
$$
We need to compute the instantaneous error $\varepsilon$ w.r.t to each the weights $u$, $v$ and bias $b$ of the neuron unit in  layer K, using the chain rule, we express the gradient as 
$$
\frac{\partial\varepsilon(n)}{\partial u_{ji}(n)} = \frac{\partial\varepsilon(n)}{\partial e_j(n)} 
\frac{\partial e_j(n)}{\partial f_j(n)}
\frac{\partial f_j(n)}{\partial m_j(n)}
\frac{\partial m_j(n)}{\partial u_{ji}(n)}\\
$$

$$
\frac{\partial\varepsilon(n)}{\partial v_{ji}(n)} = \frac{\partial\varepsilon(n)}{\partial e_j(n)} 
\frac{\partial e_j(n)}{\partial f_j(n)}
\frac{\partial f_j(n)}{\partial m_j(n)}
\frac{\partial m_j(n)}{\partial v_{ji}(n)}\\
$$

$$
\frac{\partial\varepsilon(n)}{\partial b_{ji}(n)} = \frac{\partial\varepsilon(n)}{\partial e_j(n)} 
\frac{\partial e_j(n)}{\partial f_j(n)}
\frac{\partial f_j(n)}{\partial m_j(n)}
\frac{\partial m_j(n)}{\partial b_{ji}(n)}
$$

Differentiating $\varepsilon(n)$ w.r.t $e_j(n)$ in equation $\ref{error_fun}$, we get the first chain in equation:
$$
\varepsilon(n)=\frac{1}{2}e^2_j(n) \label{error_fun}\\
\frac{\partial\varepsilon(n)}{\partial e_j(n)}  = e_j(n)
$$
Taking the derivative of error signal $e_j(n)$ w.r.t. $f_j(n)$, we get the second chain in the equation:
$$
e_j(n) = \hat{y}_j - f_j(n)\\
\frac{\partial e_j(n)}{\partial f_j(n)} = -1
$$
Taking the derivative of the activation function in equation ${\ref{sigmoid}}$, we get the third chain:
$$
\begin{align}
\frac{\partial f_j}{\partial m_j} &= \frac{1}{1+\exp(-m_j)}\cdot\frac{-\exp(-m_j)}{1+\exp(-m_j)}\\
&=f(m_j)(1-f(m_j))
\end{align}
$$
For the last chain, we get the derivatives of the weights and bias respectively:
$$
\frac{\partial m_j}{\partial u_{ji}} =\sum^{N_{k-1}}_{i=1} x^2_{k-1,i}\\
\frac{\partial m_j}{\partial v_{ji}} = \sum^{N_{k-1}}_{i=1} x_{k-1,i}\\
\frac{\partial m_j}{\partial b_{j}} = 1\\
$$
Combining the steps above:
$$
\frac{\partial\varepsilon(n)}{\partial u_{ji}(n)} =  -e_j(n)f_j(m_j)(1-f_j(m_j))x^2_{k-1,i}\\
$$

$$
\frac{\partial\varepsilon(n)}{\partial v_{ji}(n)} = \\
$$


$$
\frac{\partial\varepsilon(n)}{\partial b_{ji}(n)} = \\
$$


The local gradient $\delta_j(n)$ are as follows:
$$
\delta_j(n) = -\frac{\partial\varepsilon(n)}{\partial f_j(n)}
$$


$u_{kji}$ and $v_{kji}$ are the trainable weights and $b_{kj}$ is the bias for the layer

Please derive the BP algorithm for MLQP in both online learning and batch learning.

### Online learning

### Batch learning

[1e-3,1e-1,3,] 

## Problem 2

Please implement an on-line BP algorithm for MLQP
(you can use any programming language), train an MLQP
with one hidden layer to classify two spirals problem,
and compare the training time and decision boundaries at
three different learning rates .



## Problem 3

1. Divide the two spirals problem into four sub-problems randomly and with prior knowledge, respectively
2. Train MLQP on these sub-problems and construct two min-max modular networks
3. Compare training time an decision boundaries of the above two min-max modular networks

