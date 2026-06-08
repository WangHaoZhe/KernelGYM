# BatchNorm1d — PyTorch 2.12 documentation

- `source_id`: `pytorch_batchnorm1d`
- `url`: https://docs.pytorch.org/docs/2.12/generated/torch.nn.BatchNorm1d.html
- `category`: `pytorch_api`
- `num_chunks`: `1`
- `tags`: `pytorch, batchnorm, normalization`

## Chunk 0

- `chunk_id`: `pytorch_batchnorm1d::chunk::0`
- `token_estimate`: `724`
- `word_count`: `724`

~~~text
# BatchNorm1d

class torch.nn. BatchNorm1d ( num_features , eps = 1e-05 , momentum = 0.1 , affine = True , track_running_stats = True , device = None , dtype = None , * , bias = True ) [source]

Applies Batch Normalization over a 2D or 3D input. Method described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift . y = x − E [ x ] V a r [ x ] + ϵ ∗ γ + β y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta y = Var [ x ] + ϵ ​ x − E [ x ] ​ ∗ γ + β The mean and standard-deviation are calculated per-dimension over the mini-batches and γ \gamma γ and β \beta β are learnable parameter vectors of size C (where C is the number of features or channels of the input). By default, the elements of γ \gamma γ are set to 1 and the elements of β \beta β are set to 0. At train time in the forward pass, the variance is calculated via the biased estimator, equivalent to torch.var(input, correction=0) . However, the value stored in the moving average of the variance is calculated via the unbiased estimator, equivalent to torch.var(input, correction=1) . Also by default, during training this layer keeps running estimates of its computed mean and variance, which are then used for normalization during evaluation. The running estimates are kept with a default momentum of 0.1. If track_running_stats is set to False , this layer then does not keep running estimates, and batch statistics are instead used during evaluation time as well. Note This momentum argument is different from one used in optimizer classes and the conventional notion of momentum. Mathematically, the update rule for running statistics here is x ^ new = ( 1 − momentum ) × x ^ + momentum × x t \hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t x ^ new ​ = ( 1 − momentum ) × x ^ + momentum × x t ​ , where x ^ \hat{x} x ^ is the estimated statistic and x t x_t x t ​ is the new observed value. Because the Batch Normalization is done over the C dimension, computing statistics on (N, L) slices, it’s common terminology to call this Temporal Batch Normalization. Parameters : num_features ( int ) – number of features or channels C C C of the input eps ( float ) – a value added to the denominator for numerical stability. Default: 1e-5 momentum ( float | None ) – the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1 affine ( bool ) – a boolean value that when set to True , this module has learnable affine parameters. Default: True track_running_stats ( bool ) – a boolean value that when set to True , this module tracks the running mean and variance, and when set to False , this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None . When these buffers are None , this module always uses batch statistics. in both training and eval modes. Default: True bias ( bool ) – If set to False , the layer will not learn an additive bias (only relevant if affine is True ). Default: True Shape: Input: ( N , C ) (N, C) ( N , C ) or ( N , C , L ) (N, C, L) ( N , C , L ) , where N N N is the batch size, C C C is the number of features or channels, and L L L is the sequence length Output: ( N , C ) (N, C) ( N , C ) or ( N , C , L ) (N, C, L) ( N , C , L ) (same shape as input) Examples: >>> # With Learnable Parameters >>> m = nn . BatchNorm1d ( 100 ) >>> # Without Learnable Parameters >>> m = nn . BatchNorm1d ( 100 , affine = False ) >>> input = torch . randn ( 20 , 100 ) >>> output = m ( input )
~~~
