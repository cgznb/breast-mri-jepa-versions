"""Bounded, deterministic batch loading and metadata-bound ROI32 image caches."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import threading

import numpy as np

from .io import read_json, write_json

CACHE_SCHEMA = "registered_roi32_image_cache_v1"
RAW_CACHE_SCHEMA = "registered_roi32_raw_image_cache_v1"
IMAGE_SHAPE = (3, 32, 128, 128)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class ROI32T0LesionCache:
    """Bounded cache of real baseline ROI arrays in the existing MRI grid.

    Callers supply T0 archive paths only. Loading the ``roi`` member leaves
    image, latent, support, and all follow-up archives untouched.
    """

    def __init__(self, paths, max_open=64):
        if type(max_open) is not int or max_open < 1:
            raise ValueError("Lesion cache max_open must be a positive integer")
        self.identities = {str(Path(path).resolve()): file_identity(path) for path in paths}
        self.max_open = max_open
        self.open = OrderedDict()
        self.lock = threading.RLock()

    def read(self, path):
        source = str(Path(path).resolve())
        with self.lock:
            if source not in self.identities:
                raise ValueError("Lesion ROI requires a declared baseline archive")
            if file_identity(source) != self.identities[source]:
                raise ValueError("Baseline lesion ROI archive changed during the run")
            if source not in self.open:
                with np.load(source, allow_pickle=False) as archive:
                    if "roi" not in archive.files:
                        raise ValueError("Baseline ROI32 archive requires a real T0 roi array")
                    array = archive["roi"]
                if array.shape != (1, *IMAGE_SHAPE[1:]) or array.dtype != np.bool_:
                    raise ValueError("T0 lesion roi must be bool [1,32,128,128] on the MRI grid")
                if not array.any():
                    raise ValueError("T0 lesion roi is empty; a lesion experiment requires a real region")
                array.flags.writeable = False
                self.open[source] = array
                if len(self.open) > self.max_open:
                    self.open.popitem(last=False)
            self.open.move_to_end(source)
            return self.open[source]

    def signature(self):
        for path, identity in self.identities.items():
            if file_identity(path) != identity:
                raise ValueError("Baseline lesion ROI archive changed during the run")
        return {"schema": "registered_roi32_observed_t0_lesion_v1", "member": "roi",
                "shape": [1, *IMAGE_SHAPE[1:]], "dtype": "bool",
                "baseline_archives": list(self.identities.values())}


class ROI32ImageCache:
    def __init__(self, directory, *, normalization, phase_order, max_open=64,
                 storage_normalization=None, image_preprocessing=None):
        if type(max_open) is not int or max_open < 1:
            raise ValueError("ROI32 cache max_open must be a positive integer")
        self.directory = Path(directory).resolve()
        self.path = self.directory / "index.json"
        self.index = read_json(self.path)
        preprocessing = image_preprocessing or {"mode": "preprocessed_roi32"}
        self.raw_storage = preprocessing == {"mode": "preprocessed_roi32_raw"}
        schema = RAW_CACHE_SCHEMA if self.raw_storage else CACHE_SCHEMA
        if (self.index.get("schema") != schema or self.index.get("shape") != list(IMAGE_SHAPE)
                or self.index.get("dtype") != "float32"
                or self.index.get("image_preprocessing") != preprocessing
                or self.index.get("phase_order") != phase_order):
            raise ValueError("ROI32 cache preprocessing/normalization/shape declaration differs")
        if self.raw_storage:
            # A raw cache is shared across folds and must never bind target scale.
            if (storage_normalization is None or self.index.get("storage_image_normalization") != storage_normalization
                    or "image_normalization" in self.index):
                raise ValueError("Raw ROI32 cache storage declaration differs or contains a target scale")
        elif self.index.get("image_normalization") != normalization or storage_normalization is not None:
            raise ValueError("ROI32 cache preprocessing/normalization/shape declaration differs")
        self.entries = {}
        for entry in self.index["images"]:
            key = entry["source"]["path"]
            if key in self.entries:
                raise ValueError("Duplicate source image in ROI32 cache")
            cached = Path(entry["cached"]["path"]).resolve()
            if not cached.is_relative_to(self.directory) or cached.suffix != ".npy":
                raise ValueError("ROI32 cached images must be NPY files inside the cache directory")
            if self.raw_storage:
                foreground = Path(entry.get("foreground_cached", {}).get("path", "")).resolve()
                if not foreground.is_relative_to(self.directory) or foreground.suffix != ".npy":
                    raise ValueError("Raw ROI32 cached foreground must be an NPY file inside the cache directory")
            self.entries[key] = entry
        self.max_open = max_open
        self.arrays = OrderedDict()
        self.lock = threading.Lock()

    def read(self, source):
        key = str(Path(source).resolve())
        with self.lock:
            if key not in self.entries:
                raise ValueError("Image is absent from the declared ROI32 cache")
            entry = self.entries[key]
            kinds = ("cached", "foreground_cached") if self.raw_storage else ("cached",)
            if (file_identity(source) != entry["source"]
                    or any(file_identity(entry[kind]["path"]) != entry[kind] for kind in kinds)):
                raise ValueError("ROI32 image cache is stale: source or cached file identity changed")
            if key not in self.arrays:
                # Copy-on-write mapping is writable for torch views without changing the file.
                array = np.load(entry["cached"]["path"], mmap_mode="c", allow_pickle=False)
                if array.shape != IMAGE_SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError("ROI32 cached image must be finite float32 [3,32,128,128]")
                if self.raw_storage:
                    foreground = np.load(entry["foreground_cached"]["path"], mmap_mode="r", allow_pickle=False)
                    if (foreground.shape != IMAGE_SHAPE or foreground.dtype != np.bool_
                            or not np.array_equal(foreground, array != 0)
                            or not foreground.reshape(3, -1).any(1).all()):
                        raise ValueError("Raw ROI32 cached foreground disagrees with raw image/background")
                    self.arrays[key] = (array, foreground)
                else:
                    self.arrays[key] = array
                if len(self.arrays) > self.max_open:
                    self.arrays.popitem(last=False)
            self.arrays.move_to_end(key)
            return self.arrays[key][0] if self.raw_storage else self.arrays[key]

    def read_foreground(self, source):
        if not self.raw_storage:
            raise ValueError("Explicit foreground is available only in a raw ROI32 cache")
        # Keep raw+foreground together while holding the LRU lock; another reader
        # may evict a previously returned entry before this method acquires it.
        key = str(Path(source).resolve())
        while True:
            self.read(source)
            with self.lock:
                if key in self.arrays:
                    return self.arrays[key][1]

    def signature(self):
        if read_json(self.path) != self.index:
            raise ValueError("ROI32 cache index changed during the run")
        for entry in self.index["images"]:
            for kind in (("source", "cached", "foreground_cached") if self.raw_storage else ("source", "cached")):
                if file_identity(entry[kind]["path"]) != entry[kind]:
                    raise ValueError("ROI32 image cache is stale: file identity changed")
        return {"directory": str(self.directory), "index": copy.deepcopy(self.index)}


def prepare_image_cache(store, directory, workers=4):
    if not store.preprocessed_roi32 or store.image_cache is not None:
        raise ValueError("Cache preparation requires the original preprocessed ROI32 NPZ store")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Cache workers must be an integer from one to eight")
    visits = sorted(((str(store.resolve(visit["image"])), visit)
                     for patient in store.patients for visit in patient["visits"] if visit is not None))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (directory / "index.json").exists():
        existing = ROI32ImageCache(directory, normalization=store.image_normalization,
                                  phase_order=store.manifest["phase_order"],
                                  storage_normalization=store.storage_image_normalization,
                                  image_preprocessing=store.image_preprocessing)
        if set(existing.entries) != {source for source, _ in visits}:
            raise ValueError("ROI32 image cache population differs from the manifest")
        existing.signature()
        return {"images": len(existing.entries), "reused": True, "directory": str(directory)}

    def prepare(item):
        index, (source, visit) = item
        before = file_identity(source)
        if store.preprocessed_roi32_raw:
            image, foreground = store._read_storage_roi32(visit)
        else:
            image, _ = store._read_scan(visit)
            foreground = None
        path = directory / f"image_{index:04d}.npy"
        if path.exists():
            saved = np.load(path, mmap_mode="r", allow_pickle=False)
            if saved.shape != image.shape or saved.dtype != image.dtype or not np.array_equal(saved, image):
                raise ValueError("An existing partial cache image differs from its source")
        else:
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                os.chmod(temporary, 0o600)
                np.save(handle, image, allow_pickle=False)
            os.replace(temporary, path)
            saved = np.load(path, mmap_mode="r", allow_pickle=False)
            if not np.array_equal(saved, image):
                raise ValueError("Derived ROI32 cache image differs from its source")
        result = {"source": before, "cached": file_identity(path)}
        if foreground is not None:
            foreground_path = directory / f"foreground_{index:04d}.npy"
            if foreground_path.exists():
                saved_foreground = np.load(foreground_path, mmap_mode="r", allow_pickle=False)
                if saved_foreground.dtype != np.bool_ or not np.array_equal(saved_foreground, foreground):
                    raise ValueError("An existing partial cached foreground differs from its source")
            else:
                temporary = foreground_path.with_suffix(".npy.tmp")
                with temporary.open("wb") as handle:
                    os.chmod(temporary, 0o600)
                    np.save(handle, foreground, allow_pickle=False)
                os.replace(temporary, foreground_path)
            result["foreground_cached"] = file_identity(foreground_path)
        if file_identity(source) != before:
            raise ValueError("Source image changed while preparing its cache")
        return result

    entries = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for entry in executor.map(prepare, enumerate(visits)):
            entries.append(entry)
            if len(entries) % 250 == 0:
                print(f"Cache verified {len(entries)}/{len(visits)} images", flush=True)
    index = {"schema": RAW_CACHE_SCHEMA if store.preprocessed_roi32_raw else CACHE_SCHEMA,
             "shape": list(IMAGE_SHAPE), "dtype": "float32",
             "image_preprocessing": copy.deepcopy(store.image_preprocessing),
             "phase_order": copy.deepcopy(store.manifest["phase_order"]), "images": entries}
    if store.preprocessed_roi32_raw:
        index["storage_image_normalization"] = copy.deepcopy(store.storage_image_normalization)
    else:
        index["image_normalization"] = copy.deepcopy(store.image_normalization)
    write_json(directory / "index.json", index)
    (directory / "index.json").chmod(0o600)
    return {"images": len(entries), "reused": False, "directory": str(directory),
            "total_bytes": sum(e[k]["size"] for e in entries for k in (("cached", "foreground_cached") if store.preprocessed_roi32_raw else ("cached",))),
            "all_images_value_equal_to_npz": True, "zero_background_preserved": True}


class BatchLoader:
    """Workers consume preplanned tasks; the training sampler never advances here."""
    def __init__(self, store, *, prefetch_batches=0, workers=2, pin_memory=False,
                 future_supervision=True):
        self.store, self.depth, self.pin_memory = store, prefetch_batches, pin_memory
        self.future_supervision = future_supervision
        self.executor = ThreadPoolExecutor(max_workers=workers) if self.depth else None
        self.pending = deque()
        self.planned = False
        self.tasks = iter(())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()

    def _batch(self, tasks):
        if not self.future_supervision:
            return self.store.batch(tasks, pin_memory=self.pin_memory, future_supervision=False)
        if self.pin_memory:
            return self.store.batch(tasks, pin_memory=True)
        return self.store.batch(tasks)

    def _fill(self):
        while len(self.pending) < self.depth:
            tasks = next(self.tasks, None)
            if tasks is None:
                break
            self.pending.append((tasks, self.executor.submit(self._batch, tasks)))

    def plan(self, tasks):
        if self.pending:
            raise RuntimeError("Cannot replace a batch plan while unconsumed batches remain")
        self.tasks, self.planned = iter(tasks), True
        self._fill()

    def fetch(self, tasks):
        if self.executor is None:
            return self._batch(tasks)
        if not self.pending:
            raise RuntimeError("Prefetch plan exhausted before the consumed training cursor")
        expected, future = self.pending.popleft()
        if expected != tasks:
            raise RuntimeError("Prefetched tasks differ from the main training sampler")
        result = future.result()
        self._fill()
        return result

    def finish_epoch(self):
        if self.pending:
            raise RuntimeError("Unconsumed prefetched batches at the end of an epoch")
        self.planned = False
