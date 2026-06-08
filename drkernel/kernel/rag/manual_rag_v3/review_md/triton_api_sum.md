# triton.language.sum — Triton  documentation

- `source_id`: `triton_api_sum`
- `url`: https://triton-lang.org/main/python-api/generated/triton.language.sum.html
- `category`: `triton_api`
- `num_chunks`: `1`
- `tags`: `triton, sum, reduction`

## Chunk 0

- `chunk_id`: `triton_api_sum::chunk::0`
- `token_estimate`: `161`
- `word_count`: `161`

~~~text
# triton.language.sum

triton.language. sum ( input , axis = None , keep_dims = False , dtype : constexpr = None )

Returns the sum of all elements in the input tensor along the provided axis The reduction operation should be associative and commutative. Parameters : input ( Tensor ) – the input values axis ( int ) – the dimension along which the reduction should be done. If None, reduce all dimensions keep_dims ( bool ) – if true, keep the reduced dimensions with length 1 dtype ( tl.dtype ) – the desired data type of the returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. This is useful for preventing data overflows. If not specified, integer and bool dtypes are upcasted to tl.int32 and float dtypes are upcasted to at least tl.float32 . This function can also be called as a member function on tensor , as x.sum(...) instead of sum(x, ...) .
~~~
