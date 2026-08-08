import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import List, Literal, Sequence
from uuid import UUID, uuid4

import numpy as np
import xarray as xr
from loguru import logger
from soundevent import data
from soundevent.geometry import compute_bounds
from tqdm import tqdm

from batdetect2.core import BaseConfig
from batdetect2.outputs.formats.base import (
    make_path_relative,
    output_formatters,
)
from batdetect2.outputs.types import OutputFormatterProtocol
from batdetect2.postprocess.types import ClipDetections, Detection
from batdetect2.targets.types import TargetProtocol


class RawOutputConfig(BaseConfig):
    name: Literal["raw"] = "raw"

    include_class_scores: bool = True
    include_features: bool = True
    include_geometry: bool = True
    n_jobs: int = 1
    show_progress: bool = False


class RawFormatter(OutputFormatterProtocol[ClipDetections]):
    def __init__(
        self,
        targets: TargetProtocol,
        include_class_scores: bool = True,
        include_features: bool = True,
        include_geometry: bool = True,
        parse_full_geometry: bool = False,
        n_jobs: int = 1,
        show_progress: bool = False,
    ):
        self.targets = targets
        self.include_class_scores = include_class_scores
        self.include_features = include_features
        self.include_geometry = include_geometry
        self.parse_full_geometry = parse_full_geometry
        self.n_jobs = n_jobs
        self.show_progress = show_progress

        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")

    def format(
        self,
        predictions: Sequence[ClipDetections],
    ) -> List[ClipDetections]:
        return list(predictions)

    def save(
        self,
        predictions: Sequence[ClipDetections],
        path: data.PathLike,
        audio_dir: data.PathLike | None = None,
    ) -> None:
        path = Path(path)

        if not path.exists():
            path.mkdir(parents=True)

        for prediction in predictions:
            logger.debug(f"Saving clip predictions {prediction.clip.uuid}")
            clip = prediction.clip
            dataset = self.pred_to_xr(prediction, audio_dir)
            dataset.to_netcdf(path / f"{clip.uuid}.nc")

    def load(self, path: data.PathLike) -> List[ClipDetections]:
        path = Path(path)
        files = list(path.glob("*.nc"))

        if self.n_jobs == 1:
            return self._load_sequential(files)

        return self._load_parallel(files)

    def _load_sequential(
        self, files: Sequence[data.PathLike]
    ) -> List[ClipDetections]:

        iterable = files
        if self.show_progress:
            iterable = tqdm(files, total=len(files))

        return [self.load_single_file(filepath) for filepath in iterable]

    def _load_parallel(
        self, files: Sequence[data.PathLike]
    ) -> List[ClipDetections]:
        with Pool(self.n_jobs) as pool:
            if not self.show_progress:
                return pool.map(self.load_single_file, files)

            return list(
                tqdm(
                    pool.imap(self.load_single_file, files),
                    total=len(files),
                )
            )

    def load_single_file(self, filepath: data.PathLike) -> ClipDetections:
        logger.debug(f"Loading clip predictions {filepath}")
        clip_data = xr.load_dataset(filepath)
        return self.pred_from_xr(clip_data)

    def pred_to_xr(
        self,
        prediction: ClipDetections,
        audio_dir: data.PathLike | None = None,
    ) -> xr.Dataset:
        clip = prediction.clip
        recording = clip.recording
        num_features = 0

        if audio_dir is not None:
            recording = recording.model_copy(
                update=dict(path=make_path_relative(recording.path, audio_dir))
            )

        values = defaultdict(list)

        for pred in prediction.detections:
            detection_id = str(uuid4())

            values["detection_id"].append(detection_id)
            values["detection_score"].append(pred.detection_score)

            start_time, low_freq, end_time, high_freq = compute_bounds(
                pred.geometry
            )

            values["start_time"].append(start_time)
            values["end_time"].append(end_time)
            values["low_freq"].append(low_freq)
            values["high_freq"].append(high_freq)

            values["geometry"].append(pred.geometry.model_dump_json())

            top_class_index = int(np.argmax(pred.class_scores))
            top_class_score = float(pred.class_scores[top_class_index])
            top_class = self.targets.class_names[top_class_index]

            values["top_class"].append(top_class)
            values["top_class_score"].append(top_class_score)

            values["class_scores"].append(pred.class_scores)
            values["features"].append(pred.features)

            num_features = len(pred.features)

        data_vars = {
            "score": (["detection"], values["detection_score"]),
            "start_time": (["detection"], values["start_time"]),
            "end_time": (["detection"], values["end_time"]),
            "low_freq": (["detection"], values["low_freq"]),
            "high_freq": (["detection"], values["high_freq"]),
            "top_class": (["detection"], values["top_class"]),
            "top_class_score": (["detection"], values["top_class_score"]),
        }

        coords = {
            "detection": ("detection", values["detection_id"]),
            "clip_start": clip.start_time,
            "clip_end": clip.end_time,
            "clip_id": str(clip.uuid),
        }

        if self.include_class_scores and values["class_scores"]:
            class_scores = np.stack(values["class_scores"], axis=0)
            data_vars["class_scores"] = (
                ["detection", "classes"],
                class_scores,
            )
            coords["classes"] = ("classes", self.targets.class_names)

        if self.include_features and values["features"]:
            features = np.stack(values["features"], axis=0)
            data_vars["features"] = (["detection", "feature"], features)
            coords["feature"] = ("feature", np.arange(num_features))

        if self.include_geometry:
            data_vars["geometry"] = (["detection"], values["geometry"])

        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "recording": recording.model_dump_json(exclude_none=True),
            },
        )

    def pred_from_xr(self, dataset: xr.Dataset) -> ClipDetections:
        clip_data = dataset

        recording = data.Recording.model_validate(
            json.loads(clip_data.attrs["recording"])
        )

        clip_id = clip_data.clip_id.item()
        clip = data.Clip.model_construct(
            recording=recording,
            uuid=UUID(clip_id),
            start_time=float(clip_data.clip_start),
            end_time=float(clip_data.clip_end),
        )

        sound_events = []

        num_detections = len(clip_data.coords["detection"])

        scores = clip_data.score.data
        start_times = clip_data.start_time.data
        end_times = clip_data.end_time.data
        low_freqs = clip_data.low_freq.data
        high_freqs = clip_data.high_freq.data

        top_class_scores = clip_data.top_class_score.data
        top_class = clip_data.top_class.data

        num_classes = len(self.targets.class_names)
        class_map = dict(
            zip(
                self.targets.class_names,
                range(num_classes),
                strict=True,
            )
        )

        geometries = None
        if self.parse_full_geometry and "geometry" in clip_data:
            geometries = clip_data.geometry.data

        class_scores = None
        if "class_scores" in clip_data:
            class_scores = clip_data.class_scores.data

        features = None
        if "features" in clip_data:
            features = clip_data.features.data

        for index in range(num_detections):
            score = scores[index]

            if geometries is not None:
                geometry = data.geometry_validate(geometries[index])
            else:
                start_time = start_times[index]
                end_time = end_times[index]
                low_freq = low_freqs[index]
                high_freq = high_freqs[index]
                geometry = data.BoundingBox.model_construct(
                    coordinates=[start_time, low_freq, end_time, high_freq]
                )

            if class_scores is not None:
                class_score = class_scores[index]
            else:
                class_score = np.zeros(num_classes)
                class_index = class_map[top_class[index]]
                class_score[class_index] = top_class_scores[index]

            feats = features[index] if features is not None else np.zeros(0)

            sound_events.append(
                Detection(
                    geometry=geometry,
                    detection_score=score,
                    class_scores=class_score,
                    features=feats,
                )
            )

        return ClipDetections(
            clip=clip,
            detections=sound_events,
        )

    @output_formatters.register(RawOutputConfig)
    @staticmethod
    def from_config(config: RawOutputConfig, targets: TargetProtocol):
        return RawFormatter(
            targets,
            include_class_scores=config.include_class_scores,
            include_features=config.include_features,
            include_geometry=config.include_geometry,
            n_jobs=config.n_jobs,
            show_progress=config.show_progress,
        )
