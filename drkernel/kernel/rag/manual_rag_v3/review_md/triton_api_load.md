# triton.language.load — Triton  documentation

- `source_id`: `triton_api_load`
- `url`: https://triton-lang.org/main/python-api/generated/triton.language.load.html
- `category`: `triton_api`
- `num_chunks`: `1`
- `tags`: `triton, memory, load`

## Chunk 0

- `chunk_id`: `triton_api_load::chunk::0`
- `token_estimate`: `361`
- `word_count`: `361`

~~~text
# triton.language.load

triton.language. load ( pointer , mask = None , other = None , boundary_check = () , padding_option = '' , cache_modifier = '' , eviction_policy = '' , volatile = False , _semantic = None )

Return a tensor of data whose values are loaded from memory at location defined by pointer : If pointer is a single element pointer, a scalar is be loaded. In this case: mask and other must also be scalars, other is implicitly typecast to pointer.dtype.element_ty , and boundary_check and padding_option must be empty. If pointer is an N-dimensional tensor of pointers, an N-dimensional tensor is loaded. In this case: mask and other are implicitly broadcast to pointer.shape , other is implicitly typecast to pointer.dtype.element_ty , and boundary_check and padding_option must be empty. If pointer is a block pointer defined by make_block_ptr , a tensor is loaded. In this case: mask and other must be None , and boundary_check and padding_option can be specified to control the behavior of out-of-bound access. Parameters : pointer ( triton.PointerType , or block of dtype=triton.PointerType ) – Pointer to the data to be loaded mask (Block of triton.int1 , optional) – if mask[idx] is false, do not load the data at address pointer[idx] (must be None with block pointers) other ( Block , optional ) – if mask[idx] is false, return other[idx] . If other is None , the masked-out value is undefined. boundary_check ( tuple of ints , optional ) – tuple of integers, indicating the dimensions which should do the boundary check padding_option – should be one of {“”, “zero”, “nan”}, the padding value to use while out of bounds. “” means an undefined value. cache_modifier (str, optional, should be one of {“”, “.ca”, “.cg”, “.cv”}, where “.ca” stands for cache at all levels, “.cg” stands for cache at global level (cache in L2 and below, not L1), and “.cv” means don’t cache and fetch again. see cache operator for more details.) – changes cache option in NVIDIA PTX eviction_policy ( str , optional ) – changes eviction policy in NVIDIA PTX volatile ( bool , optional ) – changes volatile option in NVIDIA PTX
~~~
