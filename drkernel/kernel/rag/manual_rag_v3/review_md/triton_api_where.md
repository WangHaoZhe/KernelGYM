# triton.language.where — Triton  documentation

- `source_id`: `triton_api_where`
- `url`: https://triton-lang.org/main/python-api/generated/triton.language.where.html
- `category`: `triton_api`
- `num_chunks`: `1`
- `tags`: `triton, where, masking`

## Chunk 0

- `chunk_id`: `triton_api_where::chunk::0`
- `token_estimate`: `123`
- `word_count`: `123`

~~~text
# triton.language.where

triton.language. where ( condition , x , y , _semantic = None )

Returns a tensor of elements from either x or y , depending on condition . Note that x and y are always evaluated regardless of the value of condition . If you want to avoid unintended memory operations, use the mask arguments in triton.load and triton.store instead. The shape of x and y are both broadcast to the shape of condition . x and y must have the same data type. Parameters : condition ( Block of triton.bool ) – When True (nonzero), yield x, otherwise yield y. x – values selected at indices where condition is True. y – values selected at indices where condition is False.
~~~
