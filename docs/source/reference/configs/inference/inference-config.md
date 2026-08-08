# Inference config reference

`InferenceConfig` controls how files are clipped and batched during
prediction-time workflows.

Defined in `batdetect2.inference.config`.

## Top-level fields

- `compile_model`
  - compile the detector before batch prediction. This is off by default.
- `loader`
  - data-loader settings for inference.
- `clipping`
  - controls how recordings are split into clips before batching.

## `loader`

Current built-in loader field:

- `batch_size` (int, default `8`)

## `clipping`

Fields:

- `enabled` (bool)
- `duration` (float, seconds)
- `overlap` (float, seconds)
- `max_empty` (float)
- `discard_empty` (bool)

## When to override this config

Override `InferenceConfig` when:

- long recordings need different clipping behavior,
- you want to tune batch size for your hardware,
- you want to opt into runtime model compilation for repeated predictions,
- you need reproducible prediction settings across runs.

## Runtime compilation

Set `compile_model: true` to compile the detector before batch inference. This
can help when you run repeated predictions with stable input shapes. For a
single short run, the compile step can cost more time than it saves.

In Python, you can also compile explicitly with `BatDetect2API.compile()` or by
passing `compile_model=True` to `BatDetect2API.from_checkpoint(...)` or
`BatDetect2API.from_config(...)`.

## Related pages

- Tune inference clipping:
  {doc}`../../../how_to/inference/tune-inference-clipping`
- Predict CLI reference:
  {doc}`../../cli/predict`
