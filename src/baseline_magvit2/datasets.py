from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T


def _read_filelist(path: str | Path) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


@dataclass(frozen=True)
class VideoDecodeConfig:
    image_size: int
    num_frames: int
    random_time_crop: bool = True
    force_num_frames: bool = True


class VideoFileListDataset(Dataset[Tensor]):

    def __init__(self, *, filelist: str | Path, config: VideoDecodeConfig):
        super().__init__()
        self.paths = _read_filelist(filelist)
        if len(self.paths) == 0:
            raise ValueError(f"Empty filelist: {filelist}")

        self.config = config

        self.spatial_transform = T.Compose(
            [
                T.Resize(config.image_size, antialias=True),
                T.CenterCrop(config.image_size),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _decode_clip(self, path: str) -> Tensor:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            start = 0
            if self.config.random_time_crop and total_frames > self.config.num_frames:
                max_start = total_frames - self.config.num_frames
                start = int(torch.randint(0, max_start + 1, (1,)).item())
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames: list[Tensor] = []
            for _ in range(self.config.num_frames):
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
                frame = frame.float().div_(255.0)
                frame = self.spatial_transform(frame)
                frames.append(frame)

            if len(frames) == 0:
                raise RuntimeError(f"Decoded 0 frames from: {path}")

            if self.config.force_num_frames and len(frames) < self.config.num_frames:
                pad_frame = frames[-1]
                frames.extend([pad_frame] * (self.config.num_frames - len(frames)))

            clip = torch.stack(frames[: self.config.num_frames], dim=1)
            return clip
        finally:
            cap.release()

    def __getitem__(self, index: int) -> Tensor:
        path = self.paths[index]
        try:
            return self._decode_clip(path)
        except Exception:
            last_err: Exception | None = None
            for _ in range(8):
                try:
                    new_index = int(torch.randint(0, len(self.paths), (1,)).item())
                    return self._decode_clip(self.paths[new_index])
                except Exception as e:
                    last_err = e
            assert last_err is not None
            raise last_err

