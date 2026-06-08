# triton.language.clamp — Triton  documentation

- `source_id`: `triton_api_clamp`
- `url`: https://triton-lang.org/main/python-api/generated/triton.language.clamp.html
- `category`: `triton_api`
- `num_chunks`: `1`
- `tags`: `triton, clamp, activation`

## Chunk 0

- `chunk_id`: `triton_api_clamp::chunk::0`
- `token_estimate`: `99`
- `word_count`: `99`

~~~text
# triton.language.clamp

triton.language. clamp ( x , min , max , propagate_nan : constexpr = <PROPAGATE_NAN.NONE: 0> , _semantic = None )

Clamps the input tensor x within the range [min, max]. Behavior when min > max is undefined. Parameters : x ( Block ) – the input tensor min ( Block ) – the lower bound for clamping max ( Block ) – the upper bound for clamping propagate_nan ( tl.PropagateNan ) – whether to propagate NaN values. Applies only to the x tensor. If either min or max is NaN, the result is undefined. See also tl.PropagateNan
~~~
