#graphNN2DImage
Graph neural network for handwritten image classification, implemented from scratch in C++.

#Basic architecture#

The architecture we use is that of a graph neural network.
There is one neuron for each pixel (after possible initial pooling), edges correspond to $8$-directionally neighboring pixels.
For a pixel $p$ we write $z_p^k$ for the input to the node corresponding to that pixel at depth $k$ and '
$$
h_p^k = \sigma(z_p^k),
$$
where 
$$\sigma \colon \mathbb{R} \to \mathbb{R}
$$ 
is our chosen activation function. 
For weight tensors $a,b,c$ to be learned we define for all but the output layer
$$
z_p^k =  a_p^{k-1} h_p^{k-1} + b_p^{k-1} N_p^{k-1} + c_p^{k-1}.
$$
Here
$$
N_p^{k-1} = \mathrm{smax}_{q \in \mathcal{N}_p} h_q^{k-1},
$$
with $\mathcal{N}_p$ the neighboring nodes of $p$ and $\mathrm{smax}$ a smooth approximation of the maximum function.

Write $K$ for the number of possible outputs.
Then the output layer has $K$ neurons whose inputs $z_i^o$ are given by
$$
z_i^o = \sum_p w_{pi} h_p^k,
$$
where the sum ios over all pixels.
The output is then given by taking activation function followed by softmax of these.

##Bckpropagation formula##
Write $\mathcal{L}$ for a fixed loss function and 
$$
\delta_p^k = \frac{\partial \mathcal{L}}{\partial z_p^k}.
$$
It is easy derives the following backpropagation formulas:
$$
 \frac{\partial \mathcal{L}}{\partial a_p^{k-1}} = \delta_p^k h_p^{k-1}, \, 
 \frac{\partial \mathcal{L}}{\partial b_p^{k-1}} = \delta_p^k N_p^{k-1}, \,
 \frac{\partial \mathcal{L}}{\partial c_p^{k-1}} = \delta_p^k,
$$
and
$$
\delta_p^k = \sigma'(z_p^k)  \sum_{q \in \mathcal{N}_p} \delta_q^{k+1} b_q^k \frac{\partial \mathrm{smax}}{\partial \mathrm{idx}(p)}
+ \sigma'(z_p^k) \delta_p^{k+1} a_p^k,
$$
where $\mathrm{idx}(p)$ indicates that we differentiate $\smax$ in the coordinate in which $h_p^k$ is used.
