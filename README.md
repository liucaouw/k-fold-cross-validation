# k-fold-cross-validation
Example of regression with fixed bases of features.

While the hold out method is an intuitive approach to determining proper fiiting models, it suffers from an obvious flaw: 
Having been chosen at random, the points assigned to the training set may not adequately describe the original data. However,
we can easily extend and robustify the hold out method: k-fold cross-validation. Performing k-fold cross-validation is often the
most computationally expensive component in solving a general regresion problem. We provide a pseudo-code for applying k-fold
cross-validation.

**________________________________________________________________________________________________________________________**  
**Algorithm** k-fold cross-validation pseudo-code  
**Input:** Data-set $(\mathbf{x_{p}}, y_{p})_{p=1}^{P}$, k (number of folds), a range of values for M to try, and a type of basis feature  
**Split** the data into k equal (as possible) size folds  
**for** s=1...k  
&emsp; **for each** M (in the range of values to try)  
&emsp; &emsp; **1)** Train a model with M basis features on sth fold's training set  
&emsp; &emsp; **2)** Compute corresponding testing error on this fold  
**Return:** Value $M^{*}$ with lowest average testing error over all k folds 

**_________________________________________________________________________________________________________________________**

### Regression with fixed bases of features  
To perform regression using a fixed basis of features (e.g., polynomials or Fourier) it is natural to choose a degree D and transform the input data using the associated basis functions. For example, employing a degree D polynomial or Fourier basis for a scalar input, we transform each input $x_{p}$ to form an associated feature vector $\mathbf{f_{p}}=\begin{bmatrix}
x_{p} & x_{p}^{2} &...  & x_{p}^{D}
\end{bmatrix}^{T}$ or $\mathbf{f_{p}}=\begin{bmatrix}
 cos(2\pi x_{p})&sin(2\pi x_{p})  &...  &cos(2\pi Dx_{p})  & sin(2\pi Dx_{p})
\end{bmatrix}^{T}$ respectively. For higher dimensions of input N fixed basis features can be similarly used; however, the sheer number of elements involved (the length of each $\mathbf{f_{p}}$) explodes for even moderate values of N and D. In any case, once feature vectors $\mathbf{f_{p}}$ have been constructed using the data we can then determine proper weights b and $\mathbf{w}$ by minimizing the Least Squares cost function as
$$\underset{b,\mathbf{w}}{minimize} \sum_{p=1}^{P}\left ( b+\mathbf{f_{p}^{T}}\mathbf{w} -y_{p}\right )^{2}$$  
Using the compact notation $\mathbf{\widetilde{w}}=\begin{bmatrix}
b\\ \mathbf{w} 
\end{bmatrix}^{T}$ and $\widetilde{\mathbf{f_{p}}}=\begin{bmatrix}
1 & \mathbf{f_{p}}
\end{bmatrix}^{T}$ for each p we may rewrite the cost as $ g\left ( \widetilde{\mathbf{w}} \right ) $. and checking the first order condition then gives the linear system of equations $$ =\sum_{p=1}^{P}$$
