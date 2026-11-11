"""Bound encoded Seldon contexts without modifying the released backend."""

import uuid
from collections import OrderedDict
from pathlib import Path

import torch
from neuralk_oss.inference.estimator import SeldonICL


def _map(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map(item, fn) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_map(item, fn) for item in value)
    return value


def _size(value):
    sizes = []
    _map(value, lambda t: sizes.append(t.numel() * t.element_size()))
    return sum(sizes)


def _headroom():
    """Smallest available memory among this process's cgroup ancestors."""
    try:
        records = [
            line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()
        ]
        memory = next((r for r in records if "memory" in r[1].split(",")), None)
        if memory is not None:
            root = Path("/sys/fs/cgroup/memory")
            group = memory[2]
            limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            root = Path("/sys/fs/cgroup")
            group = next(r[2] for r in records if r[0] == "0" and r[1] == "")
            limit_name, usage_name = "memory.max", "memory.current"
        directory = root / group.lstrip("/")
        available = []
        while directory.is_relative_to(root):
            path = directory / limit_name
            if path.is_file():
                limit = path.read_text().strip()
                if limit != "max" and int(limit) < 2**60:
                    available.append(int(limit) - int((directory / usage_name).read_text()))
            if directory == root:
                break
            directory = directory.parent
        return min(available) if available else None
    except (OSError, StopIteration, ValueError):
        return None


class BoundedSeldonICL(SeldonICL):
    """Retain small inputs in SDM; manage large encoded contexts separately."""

    cache_budget = 60 * 1024**3
    memory_reserve = 16 * 1024**3

    def _trim(self, incoming=0):
        while self._encoded:
            available = _headroom()
            if self._encoded_bytes + incoming <= self.cache_budget and (
                available is None or incoming + self.memory_reserve <= available
            ):
                break
            _, (_, size) = self._encoded.popitem(last=False)
            self._encoded_bytes -= size

    def clear(self):
        super().clear()
        self._encoded = OrderedDict()
        self._encoded_bytes = 0

    def __getstate__(self):
        state = super().__getstate__()
        # Performance caches are rebuilt after load; do not pickle gigabytes
        # of disposable encodings or retain them in AutoGluon's refit template.
        state["_encoded"] = OrderedDict()
        state["_encoded_bytes"] = 0
        return state

    def _network(self, x, y, *, categorical_mask, cache=None):
        if cache is None:
            return super()._network(x, y, categorical_mask=categorical_mask, cache=None)
        lead, (rows, columns) = x.shape[:-2], x.shape[-2:]
        X = x.reshape(-1, rows, columns)
        mask = categorical_mask.expand(*lead, columns).reshape(-1, columns)
        if cache.is_recording:
            count = y.shape[-1]
            cache["inputs"] = (X[:, :count].clone(), y.reshape(-1, count).clone(), mask.clone())
            cache["encoding_key"] = uuid.uuid4().hex
            return x.new_zeros((*lead, 0, self._out_dim))
        key = cache["encoding_key"]
        self._trim()
        if key in self._encoded:
            stored, size = self._encoded.pop(key)
            self._encoded[key] = (stored, size)
            context = _map(stored, lambda t: t.to(X.device))
        else:
            train, labels, train_mask = cache["inputs"]
            config = self.model.config
            dtype = (
                torch.get_autocast_dtype(train.device.type)
                if torch.is_autocast_enabled(train.device.type)
                else next(self.model.parameters()).dtype
            )
            estimate = (
                train.shape[0]
                * dtype.itemsize
                * (
                    columns * config.num_inducing_points * config.embed_dim * config.col.num_blocks
                    + train.shape[1] * config.icl_width * config.icl.num_blocks
                    + columns
                )
            )
            # Make room before allocating the next context, including on CPU.
            self._trim(estimate)
            context = self.model.encode_context(train, labels, train_mask)
            size = _size(context)
            self._trim(size)
            available = _headroom()
            if size <= self.cache_budget and (
                available is None or size + self.memory_reserve <= available
            ):
                stored = _map(context, lambda t: t.detach().to("cpu", copy=True))
                self._encoded[key] = (stored, size)
                self._encoded_bytes += size
        output = self._predict(X, context, mask)
        return output.view(*lead, *output.shape[-2:])
