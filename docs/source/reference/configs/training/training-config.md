# Training config reference

`TrainingConfig` controls the training loop, optimisation, data loading, losses,
and validation tasks.

Defined in `batdetect2.train.config`.

## Top-level fields

- `compile_model`
  - compile the detector before training starts. This is off by default.
- `train_loader`
  - training data loading and clipping settings.
- `val_loader`
  - validation data loading and clipping settings.
- `optimizer`
  - optimiser type and learning rate settings.
- `scheduler`
  - learning-rate schedule settings.
- `loss`
  - detection, classification, and size loss settings.
- `trainer`
  - PyTorch Lightning trainer settings such as `max_epochs`.
- `labels`
  - target label generation settings.
- `validation`
  - evaluation tasks used during validation.
- `checkpoints`
  - checkpoint saving settings.

## What this config controls

Use `TrainingConfig` when you want to change things like:

- batch size,
- augmentation,
- optimiser and scheduler settings,
- runtime options such as model compilation,
- number of epochs,
- validation frequency,
- checkpoint behaviour.

## Runtime options

Use `compile_model: true` to call `torch.compile` on the detector used during
training. This can help on longer runs with stable tensor shapes, but it may be
slower for short CPU-only experiments because PyTorch has to compile the graph
before it can reuse it.

Example files live under `example_data/configs/`, including
`example_data/configs/training.yaml`.

## Related pages

- Evaluation config:
  {doc}`../evaluation/evaluation-config`
- Train command reference:
  {doc}`../../cli/train`
- Fine-tune from a checkpoint:
  {doc}`../../../how_to/training/fine-tune-from-a-checkpoint`
