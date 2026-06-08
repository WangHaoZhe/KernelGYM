# triton.language.max — Triton  documentation

- `source_id`: `triton_api_max`
- `url`: https://triton-lang.org/main/python-api/generated/triton.language.max.html
- `category`: `triton_api`
- `num_chunks`: `1`
- `tags`: `triton, max, reduction`

## Chunk 0

- `chunk_id`: `triton_api_max::chunk::0`
- `token_estimate`: `151`
- `word_count`: `151`

~~~text
# triton.language.max

triton.language. max ( input , axis = None , return_indices = False , return_indices_tie_break_left = True , keep_dims = False )

Returns the maximum of all elements in the input tensor along the provided axis The reduction operation should be associative and commutative. Parameters : input ( Tensor ) – the input values axis ( int ) – the dimension along which the reduction should be done. If None, reduce all dimensions keep_dims ( bool ) – if true, keep the reduced dimensions with length 1 return_indices ( bool ) – if true, return index corresponding to the maximum value return_indices_tie_break_left ( bool ) – if true, in case of a tie (i.e., multiple elements have the same maximum value), return the left-most index for values that aren’t NaN This function can also be called as a member function on tensor , as x.max(...) instead of max(x, ...) .
~~~
