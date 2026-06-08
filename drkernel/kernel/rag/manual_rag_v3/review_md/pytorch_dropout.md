# Dropout — PyTorch 2.12 documentation

- `source_id`: `pytorch_dropout`
- `url`: https://docs.pytorch.org/docs/2.12/generated/torch.nn.Dropout.html
- `category`: `pytorch_api`
- `num_chunks`: `1`
- `tags`: `pytorch, dropout, regularization`

## Chunk 0

- `chunk_id`: `pytorch_dropout::chunk::0`
- `token_estimate`: `232`
- `word_count`: `232`

~~~text
# Dropout

class torch.nn. Dropout ( p = 0.5 , inplace = False ) [source]

During training, randomly zeroes some of the elements of the input tensor with probability p . The zeroed elements are chosen independently for each forward call and are sampled from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call. This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper Improving neural networks by preventing co-adaptation of feature detectors . Furthermore, the outputs are scaled by a factor of 1 1 − p \frac{1}{1-p} 1 − p 1 ​ during training. This means that during evaluation the module simply computes an identity function. Parameters : p ( float ) – probability of an element to be zeroed. Default: 0.5 inplace ( bool ) – If set to True , will do this operation in-place. Default: False Shape: Input: ( ∗ ) (*) ( ∗ ) . Input can be of any shape Output: ( ∗ ) (*) ( ∗ ) . Output is of the same shape as input Examples: >>> m = nn . Dropout ( p = 0.2 ) >>> input = torch . randn ( 20 , 16 ) >>> output = m ( input ) forward ( input ) [source] Runs the forward pass. Return type : Tensor
~~~
