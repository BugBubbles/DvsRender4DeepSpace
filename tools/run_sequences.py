"""Prepare or render the ten supplied native-camera sequences, with a shared cap.

No simulator source, executables or logs are written into the dataset directory.
The optimized renderer retains native camera/spatial quadrature and circuit cells;
per-pixel optical time refinement is recorded in numerical_model.json.
"""
from pathlib import Path
import argparse
import concurrent.futures
import csv
import json
import math
import os
import subprocess
import threading
import time
import h5py
import numpy as np
import yaml
from dataset_storage import DatasetBudget, DatasetSizeLimit, HARD_LIMIT, storage_report

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = PROJECT.parent / 'audit/2026-10-04/sequences'


def estimated_size(manifest, root):
    total = storage_report(root)['by_kind_bytes'].get('maps', 0)
    components = []
    for seq in manifest['sequences']:
        directory = root / 'sequences' / seq['sequence_id']
        cfg = yaml.safe_load((directory / 'source_config.yaml').read_text())
        bandwidth = float(cfg['physic']['readout_bandwidth'])
        if bandwidth <= 0:
            raise RuntimeError('no finite readout bound for dataset volume estimate')
        duration = seq['duration_seconds']
        maximum_events = math.ceil(duration * bandwidth) + int(cfg['physic']['readout_queue_capacity']) + 1
        frames = math.ceil(duration * cfg['scene']['cam_fps']) + 1
        raw_frames = frames * cfg['intrinsic']['width'] * cfg['intrinsic']['height'] * 5
        # No dependency on image compressibility: 13 bytes per event, uint8
        # intensity and float32 depth, plus a conservative metadata allowance.
        bytes_bound = maximum_events * 13 + raw_frames + 32 * 1024 * 1024
        components.append({'sequence': seq['sequence_id'], 'maximum_events': maximum_events,
                           'maximum_frames': frames, 'uncompressed_data_bound_bytes': bytes_bound})
        total += bytes_bound
    return {'assumptions': 'enabled configured FIFO bandwidth, at most queue_capacity transport-tail events; no raw photon/ray/debug buffers',
            'uncompressed_dataset_bound_bytes': total,
            'sequences': components}


def validate_sequence(directory):
    cfg = yaml.safe_load((directory / 'source_config.yaml').read_text())
    camera = cfg['intrinsic']
    calib = yaml.safe_load((directory / 'calibration.yaml').read_text())
    if calib['resolution'] != [camera['width'], camera['height']]:
        raise RuntimeError('camera calibration dimensions disagree with source config')
    for v, expected in zip([calib['K'][0][0], calib['K'][1][1], calib['K'][0][2], calib['K'][1][2]],
                           [camera['fx'], camera['fy'], camera['cx'], camera['cy']]):
        if abs(v - expected) > 1e-8:
            raise RuntimeError('calibration intrinsics disagree with source config')
    counts = {}
    poses = None
    for name in ['poses', 'motion', 'pose_geographic']:
        with (directory / (name + '.csv')).open() as inp:
            rows = list(csv.reader(inp))
        data = np.asarray([[float(x) for x in row] for row in rows[1:]], dtype=np.float64)
        expected_rows = math.ceil(60 * cfg['scene']['imu_fps']) + 1
        if len(data) != expected_rows or data[0, 0] != 0 or data[-1, 0] != 60 or not np.isfinite(data).all():
            raise RuntimeError('incomplete 60 s pose or geographic export')
        if not (np.diff(data[:, 0]) > 0).all():
            raise RuntimeError('unordered pose times')
        counts[name + '_rows'] = len(data)
        if name == 'poses':
            poses = data
            if not np.allclose(np.linalg.norm(data[:, 4:8], axis=1), 1, atol=1e-5):
                raise RuntimeError('invalid pose quaternion')
        if name == 'motion':
            speed = np.linalg.norm(data[:, 1:4], axis=1)
            counts['median_linear_speed_m_s'] = float(np.median(speed))
            counts['maximum_linear_speed_m_s'] = float(speed.max())
        if name == 'pose_geographic':
            seq_meta = json.loads((directory / 'sequence.json').read_text())
            reference = json.loads((directory / seq_meta['map_reference']).read_text())
            origin = np.asarray(reference['origin_projected_m'] + [reference['origin_altitude_source_m']])
            if not np.allclose(data[:, 1:4], poses[:, 1:4] + origin, rtol=0, atol=1e-5):
                raise RuntimeError('geographic positions do not match local poses and map origin')
            if (np.abs(data[:, 5]) > 90).any():
                raise RuntimeError('invalid lunar latitude')
    rendered = (directory / 'events.h5').exists() or (directory / 'frames.h5').exists()
    if not rendered:
        counts['status'] = 'poses_prepared_not_rendered'
        return counts
    progress = json.loads((directory / 'render_progress.json').read_text())
    if progress['status'] != 'rendered_pending_validation':
        raise RuntimeError('renderer has not finished the full observation interval')
    with h5py.File(directory / 'frames.h5', 'r') as f:
        expected_frames = math.ceil(60 * cfg['scene']['cam_fps']) + 1
        shape = (expected_frames, camera['height'], camera['width'])
        if f['images'].shape != shape or f['depth'].shape != shape:
            raise RuntimeError('missing frames or native camera dimensions changed')
        ts = f['t'][:]
        expected_times = np.round(np.minimum(np.arange(expected_frames) / cfg['scene']['cam_fps'], 60) * 1e6).astype('int64')
        if not np.array_equal(ts, expected_times):
            raise RuntimeError('frame timestamps do not cover native sampling from 0 to 60 s')
        hits = total = 0
        for i in range(expected_frames):
            depth = f['depth'][i]
            if not np.isfinite(depth).all() or ((depth < 0) & (depth != -1)).any():
                raise RuntimeError('invalid depth samples')
            hits += int((depth >= 0).sum()); total += depth.size
        counts.update(frames=expected_frames, hit_depth_fraction=hits / total)
    with h5py.File(directory / 'events.h5', 'r') as f:
        n = len(f['t'])
        if any(len(f[key]) != n for key in ['x', 'y', 'p']):
            raise RuntimeError('event columns differ in length')
        last = -1; on = 0
        for first in range(0, n, 1 << 20):
            sl = slice(first, min(n, first + (1 << 20)))
            t, x, y, p = [f[key][sl] for key in ['t', 'x', 'y', 'p']]
            if t[0] < last or (np.diff(t) < 0).any() or t[0] < 0:
                raise RuntimeError('unordered or negative event timestamps')
            if ((x < 0) | (x >= camera['width']) | (y < 0) | (y >= camera['height'])).any():
                raise RuntimeError('event coordinate exceeds native camera bounds')
            if ((p != 0) & (p != 1)).any():
                raise RuntimeError('invalid event polarity')
            last = int(t[-1]);on += int(p.sum())
        if last > math.ceil(progress['event_transport_end_seconds'] * 1e6):
            raise RuntimeError('events extend beyond the recorded readout tail')
        idx = f['ms_to_idx'][:]
        if len(idx) < 60001 or (np.diff(idx) < 0).any() or (idx < 0).any() or (idx > n).any():
            raise RuntimeError('invalid or incomplete event millisecond index')
        counts.update(events=n, on_events=on, off_events=n-on,
                      readout_dropped_events=progress['readout_dropped_events'])
    counts['status'] = 'complete'
    return counts


class BuildGate:
    """Limit host mesh builds; release as soon as the GPU owns the scene."""
    def __init__(self, semaphore=None):
        self.semaphore = semaphore
        self.held = False
    def __enter__(self):
        if self.semaphore is not None:
            self.semaphore.acquire()
            self.held = True
        return self
    def release(self):
        if self.held:
            self.held = False
            self.semaphore.release()
    def __exit__(self, *args):
        self.release()


def run(root, audit, mode, gpus):
    root = root.resolve();audit.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / 'dataset.json').read_text())
    budget = DatasetBudget(root)
    size = estimated_size(plan, root)
    (audit / 'size_estimate.json').write_text(json.dumps(size, indent=2) + '\n')
    if mode == 'plan':
        print(json.dumps({'mode': mode, **storage_report(root), **size}, indent=2))
        return
    stopped = threading.Event();large_scene = threading.Semaphore(2);metadata_lock = threading.Lock()

    def persist(path, value):
        try:
            budget.write_json(path, value)
        except DatasetSizeLimit:
            # The 20 GB gap is also the emergency allowance for tiny completion
            # metadata after every data producer has been stopped.
            text = json.dumps(value, indent=2) + '\n'
            if len(text.encode()) > 1024 * 1024:
                raise RuntimeError('emergency completion metadata is too large')
            DatasetBudget(root, stop=HARD_LIMIT).write_text(path, text)

    def job(seq, gpu):
        if stopped.is_set():
            return
        directory = root / 'sequences' / seq['sequence_id']
        if mode in ('render-native', 'render-optimized') and seq['status'] == 'complete':
            validate_sequence(directory)
            return
        dem = root / seq['resources']['dem']
        albedo = str(root / seq['resources']['albedo']) if 'albedo' in seq['resources'] else ''
        try:
            with budget.reserve(8 * 1024 * 1024):
                pass
        except DatasetSizeLimit as exc:
            stopped.set()
            with metadata_lock:
                seq['status']='stopped_size_limit';seq['stop_reason']=str(exc)
                persist(directory / 'sequence.json',seq)
            return
        command = [str(PROJECT / 'bin/sequence')]
        if mode == 'prepare-poses':
            command.append('--prepare-poses')
        elif mode == 'render-optimized':
            command.append('--optimized')
        command += [str(root), str(directory), str(dem), albedo]
        env = os.environ.copy();env['CUDA_VISIBLE_DEVICES'] = str(gpu);env['OMP_NUM_THREADS'] = '16'
        log = audit / (seq['sequence_id'] + ('.prepare.log' if mode == 'prepare-poses' else '.render.log'))
        build_limit = large_scene if seq['source_scene'] == 'southpole' and mode in ('render-native', 'render-optimized') else None
        with BuildGate(build_limit) as gate:
            if stopped.is_set():return
            if mode in ('render-native', 'render-optimized'):
                with metadata_lock:
                    seq['status'] = 'rendering';persist(directory / 'sequence.json', seq);persist(root / 'dataset.json', plan)
            with log.open('w') as out:
                process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env)
                reader = log.open('r');log_tail = ''
                while process.poll() is None:
                    log_tail += reader.read()
                    if 'SCENE_GPU_READY' in log_tail:
                        gate.release()
                    log_tail = log_tail[-64:]
                    if stopped.wait(1):
                        process.terminate()
                        try:process.wait(timeout=30)
                        except subprocess.TimeoutExpired:process.kill();process.wait()
                        break
                reader.close()
            if process.returncode == 42:
                stopped.set();seq['status'] = 'stopped_size_limit'
            elif stopped.is_set():
                seq['status'] = 'interrupted_after_batch_stop'
            elif process.returncode:
                seq['status'] = 'failed';seq['failure_log'] = str(log);stopped.set()
            else:
                try:
                    result = validate_sequence(directory)
                    seq['status'] = result['status']
                    persist(directory / 'validation.json', result)
                except Exception as exc:
                    seq['status'] = 'failed_validation';seq['failure_reason'] = str(exc);stopped.set()
            with metadata_lock:
                persist(directory / 'sequence.json', seq)
                persist(root / 'dataset.json', plan)

    if mode in ('render-native', 'render-optimized'):
        plan['status'] = 'rendering'
        plan['started_unix_seconds'] = time.time()
        persist(root / 'dataset.json', plan)

    # Each GPU runs at most one process; bound the 330 M-vertex southpole
    # scene to two simultaneous host builds; rendering continues on all GPUs.
    queues = [plan['sequences'][i::len(gpus)] for i in range(len(gpus))]
    def worker(i):
        for seq in queues[i]:
            if stopped.is_set():break
            job(seq, gpus[i])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        try:
            list(executor.map(worker, range(len(gpus))))
        except BaseException:
            stopped.set()
            raise
    report = storage_report(root);report['estimate'] = size
    (audit / 'storage_report.json').write_text(json.dumps(report, indent=2) + '\n')
    statuses = [seq['status'] for seq in plan['sequences']]
    if all(s == 'complete' for s in statuses):plan['status'] = 'complete'
    elif all(s == 'poses_prepared_not_rendered' for s in statuses):plan['status'] = 'poses_prepared_not_rendered'
    elif stopped.is_set():plan['status'] = 'stopped_size_limit' if 'stopped_size_limit' in statuses or report['stop_required'] else 'failed_or_interrupted'
    else:plan['status'] = 'incomplete'
    plan['logical_bytes_at_last_check'] = report['logical_bytes']
    persist(root / 'dataset.json', plan)
    print(json.dumps({'status': plan['status'], 'logical_bytes': report['logical_bytes'], 'sequences': statuses}))
    if stopped.is_set():raise SystemExit(42 if plan['status'] == 'stopped_size_limit' else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/temp/cbf_datasets/simulate'))
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--mode', choices=['plan', 'prepare-poses', 'render-native', 'render-optimized'], default='plan')
    parser.add_argument('--gpus', default='0')
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or min(gpus) < 0 or len(gpus) != len(set(gpus)):
        parser.error('provide distinct nonnegative GPU indices')
    run(args.root, args.audit, args.mode, gpus)
