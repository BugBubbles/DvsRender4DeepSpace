"""Aggregate storage accounting shared by preparation and render supervision."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import fcntl
import json
import os
import shutil
import tempfile

HARD_LIMIT = 500_000_000_000
STOP_LIMIT = 480_000_000_000


class DatasetSizeLimit(RuntimeError):
    pass


def storage_report(root):
    root = Path(root)
    by_kind = {}
    logical = allocated = count = 0
    largest = []
    for p in root.rglob('*'):
        if p.is_symlink():
            raise RuntimeError(f'dataset symlinks are unsupported for capacity accounting: {p}')
        if not p.is_file():
            continue
        stat = p.stat()
        logical += stat.st_size
        allocated += stat.st_blocks * 512
        count += 1
        relative = p.relative_to(root)
        if relative.parts[0] == 'maps':
            kind = 'maps'
        elif p.name.startswith('events'):
            kind = 'events'
        elif p.name.startswith('frames'):
            kind = 'images_and_depth'
        elif p.suffix == '.csv':
            kind = 'poses_and_motion'
        else:
            kind = 'calibration_and_metadata'
        by_kind[kind] = by_kind.get(kind, 0) + stat.st_size
        largest.append({'file': str(relative), 'logical_bytes': stat.st_size})
    return {'root': str(root), 'logical_bytes': logical,
            'allocated_bytes': allocated, 'file_count': count,
            'by_kind_bytes': by_kind,
            'largest_files': sorted(largest, key=lambda x: x['logical_bytes'], reverse=True)[:15],
            'hard_limit_bytes': HARD_LIMIT, 'stop_before_bytes': STOP_LIMIT,
            'limit_accounting': 'sum of all apparent file sizes, including maps and partial files',
            'stop_required': logical >= STOP_LIMIT}


class DatasetBudget:
    def __init__(self, root, hard=HARD_LIMIT, stop=STOP_LIMIT):
        self.root = Path(root).resolve(strict=True)
        if not 0 < stop <= hard <= HARD_LIMIT:
            raise ValueError('invalid dataset capacity limits')
        self.hard, self.stop = hard, stop
        # Match the C++ DatasetBudget lock name exactly.
        key = 14695981039346656037
        for c in str(self.root).encode('utf8'):
            key = ((key ^ c) * 1099511628211) & ((1 << 64) - 1)
        self.lock = Path(tempfile.gettempdir()) / f'dvs-dataset-{key:x}.lock'

    def require_path(self, path):
        Path(path).resolve().relative_to(self.root)

    @contextmanager
    def reserve(self, max_growth):
        if max_growth < 0:
            raise ValueError('negative storage reservation')
        descriptor = os.open(self.lock, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            before = storage_report(self.root)['logical_bytes']
            if before >= self.stop or before + max_growth > self.stop:
                raise DatasetSizeLimit(f'dataset capacity stop: current={before}, '
                                      f'reserved_write={max_growth}, stop={self.stop}, hard={self.hard} bytes')
            if shutil.disk_usage(self.root).free < max_growth + 128 * 1024 * 1024:
                raise DatasetSizeLimit('insufficient filesystem space for reserved dataset write')
            yield
            after = storage_report(self.root)['logical_bytes']
            if after > before + max_growth:
                raise DatasetSizeLimit('writer exceeded its storage reservation; batch must stop')
            if after > self.hard:
                raise DatasetSizeLimit('dataset hard capacity limit reached; batch must stop')
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def write_text(self, path, text):
        path = Path(path)
        self.require_path(path)
        data = text.encode('utf8')
        with self.reserve(len(data)):
            partial = path.with_name(path.name + '.partial')
            with partial.open('wb') as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            partial.replace(path)

    def write_json(self, path, value):
        self.write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/temp/cbf_datasets/simulate')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = storage_report(Path(args.root))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
