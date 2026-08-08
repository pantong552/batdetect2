from pathlib import Path

import pytest
from soundevent import data

from batdetect2.api_v2 import BatDetect2API
from batdetect2.inference import InferenceConfig
from batdetect2.inference.batch import run_batch_inference
from batdetect2.targets import build_roi_mapping, build_targets
from batdetect2.train import load_model_from_checkpoint
from tests.utils import assert_clip_detections_equal

pytestmark = pytest.mark.slow


def test_run_batch_inference_matches_single_clip_inference(
    contrib_dir: Path,
) -> None:
    recording = data.Recording.from_file(
        contrib_dir / "jeff37" / "0166_20240531_223911.wav"
    )
    clips = [
        data.Clip(recording=recording, start_time=start, end_time=start + 1.0)
        for start in (0.0, 1.0, 2.0)
    ]
    model, configs = load_model_from_checkpoint()
    targets = build_targets(configs.targets)
    roi_mapper = build_roi_mapping(configs.targets.roi)

    batched_predictions = run_batch_inference(
        model,
        clips,
        targets=targets,
        roi_mapper=roi_mapper,
        batch_size=3,
        num_workers=0,
    )
    single_predictions = [
        run_batch_inference(
            model,
            [clip],
            targets=targets,
            roi_mapper=roi_mapper,
            batch_size=1,
            num_workers=0,
        )[0]
        for clip in clips
    ]

    assert len(batched_predictions) == len(single_predictions)

    for batched, single in zip(
        batched_predictions,
        single_predictions,
        strict=True,
    ):
        assert_clip_detections_equal(batched, single)


def test_run_batch_inference_compiles_detector_when_config_requests_compile(
    example_annotations: list[data.ClipAnnotation],
    record_detector_compilation,
) -> None:
    api = BatDetect2API.from_config()
    recorder = record_detector_compilation(api.model)

    predictions = run_batch_inference(
        api.model,
        [example_annotations[0].clip],
        targets=api.targets,
        roi_mapper=api.roi_mapper,
        audio_loader=api.audio_loader,
        preprocessor=api.preprocessor,
        output_transform=api.output_transform,
        inference_config=InferenceConfig(compile_model=True),
        batch_size=1,
        num_workers=0,
    )

    assert predictions
    assert recorder.compile_count == 1
    assert recorder.call_count > 0


def test_run_batch_inference_does_not_recompile_compiled_detector(
    example_annotations: list[data.ClipAnnotation],
    record_detector_compilation,
) -> None:
    api = BatDetect2API.from_config()
    recorder = record_detector_compilation(api.model)
    api.compile()

    predictions = run_batch_inference(
        api.model,
        [example_annotations[0].clip],
        targets=api.targets,
        roi_mapper=api.roi_mapper,
        audio_loader=api.audio_loader,
        preprocessor=api.preprocessor,
        output_transform=api.output_transform,
        inference_config=InferenceConfig(compile_model=True),
        batch_size=1,
        num_workers=0,
    )

    assert predictions
    assert recorder.compile_count == 1
    assert recorder.call_count > 0
