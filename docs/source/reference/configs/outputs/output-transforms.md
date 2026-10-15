# Output transforms reference

Output transforms operate after decoding and before formatting.

Defined in `batdetect2.outputs.transforms`.

## Top-level config

`OutputTransformConfig` contains:

- `detection_transforms`
- `clip_transforms`

## Detection transforms

Detection transforms operate on one detection at a time.

Built-in examples include:

- filtering by frequency,
- filtering by duration.
- masking class scores when a detection's frequency is outside that class's
  configured low- and high-frequency ranges.

Filtering transforms can remove detections entirely. A class-frequency mask
instead preserves the detection and sets disallowed class scores to zero.
Classes omitted from `class_ranges` remain unrestricted. If every class is
masked, the detection is retained without a species label.

```yaml
transform:
  detection_transforms:
    - name: mask_classes_by_frequency
      class_ranges:
        nycpla:
          low_freq:
            min: 18920
            max: 22000
          high_freq:
            min: 19000
            max: 25650
  clip_transforms: []
```

The range keys must match class names in the targets configuration, and all
frequency values are in Hz. Both the predicted low-frequency boundary and the
predicted high-frequency boundary must be inside their respective ranges.
Choose the bounds using reviewed data and validate their effect before batch
inference. This is an inference-time rule and does not change model training.

## Clip transforms

Clip transforms operate on the list of detections for one clip.

Built-in examples include:

- removing detections above Nyquist,
- removing detections at clip edges.

## Related pages

- Outputs config:
  {doc}`outputs-config`
- Understanding outputs:
  {doc}`../../../explanation/interpreting-formatted-outputs`
