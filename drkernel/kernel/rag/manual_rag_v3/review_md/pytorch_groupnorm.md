# GroupNorm — PyTorch 2.12 documentation

- `source_id`: `pytorch_groupnorm`
- `url`: https://docs.pytorch.org/docs/2.12/generated/torch.nn.GroupNorm.html
- `category`: `pytorch_api`
- `num_chunks`: `1`
- `tags`: `pytorch, groupnorm, normalization`

## Chunk 0

- `chunk_id`: `pytorch_groupnorm::chunk::0`
- `token_estimate`: `428`
- `word_count`: `428`

~~~text
# GroupNorm

class torch.nn. GroupNorm ( num_groups , num_channels , eps = 1e-05 , affine = True , device = None , dtype = None , * , bias = True ) [source]

Applies Group Normalization over a mini-batch of inputs. This layer implements the operation as described in the paper Group Normalization y = x − E [ x ] V a r [ x ] + ϵ ∗ γ + β y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta y = Var [ x ] + ϵ ​ x − E [ x ] ​ ∗ γ + β The input channels are separated into num_groups groups, each containing num_channels / num_groups channels. num_channels must be divisible by num_groups . The mean and standard-deviation are calculated separately over each group. γ \gamma γ and β \beta β are learnable per-channel affine transform parameter vectors of size num_channels if affine is True . The variance is calculated via the biased estimator, equivalent to torch.var(input, correction=0) . This layer uses statistics computed from input data in both training and evaluation modes. Parameters : num_groups ( int ) – number of groups to separate the channels into num_channels ( int ) – number of channels expected in input eps ( float ) – a value added to the denominator for numerical stability. Default: 1e-5 affine ( bool ) – a boolean value that when set to True , this module has learnable per-channel affine parameters initialized to ones (for weights) and zeros (for biases). Default: True bias ( bool ) – If set to False , the layer will not learn an additive bias (only relevant if affine is True ). Default: True Shape: Input: ( N , C , ∗ ) (N, C, *) ( N , C , ∗ ) where C = num_channels C=\text{num\_channels} C = num_channels Output: ( N , C , ∗ ) (N, C, *) ( N , C , ∗ ) (same shape as input) Examples: >>> input = torch . randn ( 20 , 6 , 10 , 10 ) >>> # Separate 6 channels into 3 groups >>> m = nn . GroupNorm ( 3 , 6 ) >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm) >>> m = nn . GroupNorm ( 6 , 6 ) >>> # Put all 6 channels into a single group (equivalent with LayerNorm) >>> m = nn . GroupNorm ( 1 , 6 ) >>> # Activating the module >>> output = m ( input )
~~~
