from pathlib import Path

import numpy as np
import pytest
from soundevent import data

from batdetect2.outputs.formats import RawOutputConfig, build_output_formatter
from batdetect2.postprocess.types import ClipDetections, Detection
from batdetect2.targets.types import TargetProtocol


@pytest.fixture
def sample_formatter(sample_targets: TargetProtocol):
    return build_output_formatter(
        config=RawOutputConfig(),
        targets=sample_targets,
    )


def test_roundtrip(
    sample_formatter,
    clip: data.Clip,
    sample_targets: TargetProtocol,
    tmp_path: Path,
):
    detections = [
        Detection(
            geometry=data.BoundingBox(
                coordinates=list(np.random.uniform(size=[4]))
            ),
            detection_score=0.5,
            class_scores=np.random.uniform(
                size=len(sample_targets.class_names)
            ),
            features=np.random.uniform(size=32),
        )
        for _ in range(10)
    ]

    prediction = ClipDetections(clip=clip, detections=detections)

    path = tmp_path / "predictions"

    sample_formatter.save(predictions=[prediction], path=path)

    recovered = sample_formatter.load(path=path)

    assert len(recovered) == 1
    assert recovered[0].clip == prediction.clip

    for recovered_prediction, detection in zip(
        recovered[0].detections,
        detections,
        strict=True,
    ):
        assert (
            recovered_prediction.detection_score == detection.detection_score
        )
        assert (
            recovered_prediction.class_scores == detection.class_scores
        ).all()
        assert (recovered_prediction.features == detection.features).all()
        assert recovered_prediction.geometry == detection.geometry


def test_roundtrip_recovers_recording_metadata(
    sample_formatter,
    create_recording,
    create_clip,
    sample_targets: TargetProtocol,
    tmp_path: Path,
):
    recording = create_recording(
        tags=[data.Tag(key="source", value="test-recorder")],
        duration=2,
        samplerate=384_000,
        time_expansion=10,
    )
    clip = create_clip(recording=recording, start_time=0.25, end_time=0.75)
    detection = Detection(
        geometry=data.BoundingBox(
            coordinates=[0.3, 45_000, 0.4, 70_000],
        ),
        detection_score=0.5,
        class_scores=np.ones(len(sample_targets.class_names)),
        features=np.ones(32),
    )
    prediction = ClipDetections(clip=clip, detections=[detection])

    path = tmp_path / "predictions"

    sample_formatter.save(predictions=[prediction], path=path)
    recovered = sample_formatter.load(path=path)

    assert len(recovered) == 1
    assert recovered[0].clip.recording.model_dump(mode="json") == (
        recording.model_dump(mode="json")
    )


def test_roundtrip_empty_detections(
    sample_formatter,
    clip: data.Clip,
    tmp_path: Path,
):
    prediction = ClipDetections(clip=clip, detections=[])

    path = tmp_path / "predictions"

    sample_formatter.save(predictions=[prediction], path=path)
    recovered = sample_formatter.load(path=path)

    assert len(recovered) == 1
    assert recovered[0].detections == []
    assert recovered[0].clip.uuid == prediction.clip.uuid
    assert recovered[0].clip.start_time == prediction.clip.start_time
    assert recovered[0].clip.end_time == prediction.clip.end_time


def test_roundtrip_loads_with_multiprocessing(
    clip: data.Clip,
    sample_targets: TargetProtocol,
    tmp_path: Path,
):
    save_formatter = build_output_formatter(
        config=RawOutputConfig(),
        targets=sample_targets,
    )
    load_formatter = build_output_formatter(
        config=RawOutputConfig(n_jobs=2),
        targets=sample_targets,
    )
    predictions = [
        ClipDetections(
            clip=data.Clip(
                recording=clip.recording,
                start_time=index,
                end_time=index + 0.5,
            ),
            detections=[
                Detection(
                    geometry=data.BoundingBox(
                        coordinates=[index, 45_000, index + 0.1, 70_000],
                    ),
                    detection_score=0.5,
                    class_scores=np.ones(len(sample_targets.class_names)),
                    features=np.ones(32),
                )
            ],
        )
        for index in range(2)
    ]

    path = tmp_path / "predictions"

    save_formatter.save(predictions=predictions, path=path)
    recovered = load_formatter.load(path=path)

    assert len(recovered) == len(predictions)
    assert {item.clip.uuid for item in recovered} == {
        item.clip.uuid for item in predictions
    }
