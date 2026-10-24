"""Prepare data-only manifests and native camera/map calibration for the render batch."""
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,os
import yaml
from dataset_storage import DatasetBudget, HARD_LIMIT, storage_report
ROOT=Path('/home/chenbofei/Blender');PROJECT=ROOT/'dvs_render'
LIMIT=HARD_LIMIT

def occupied(root):
    return storage_report(root)['logical_bytes']

def checksum(path):
    digest=hashlib.sha256()
    with path.open('rb') as inp:
        for block in iter(lambda:inp.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def copy_guarded(source,destination,root):
    if destination.exists():
        if source.stat().st_size==destination.stat().st_size and checksum(source)==checksum(destination):return
        raise RuntimeError(f'refusing to replace mismatched existing dataset resource {destination}')
    budget=DatasetBudget(root)
    budget.require_path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_suffix(destination.suffix+'.partial')
    with budget.reserve(source.stat().st_size):
        with source.open('rb') as inp,partial.open('wb') as out:
            shutil.copyfileobj(inp,out,8*1024*1024)
            out.flush();os.fsync(out.fileno())
        if checksum(source)!=checksum(partial):raise RuntimeError(f'resource copy checksum mismatch {source}')
        partial.rename(destination)

def prepare(destination):
    destination.mkdir(parents=True,exist_ok=True)
    budget=DatasetBudget(destination)
    previous=destination/'dataset.json'
    if previous.exists() and json.loads(previous.read_text()).get('status')!='prepared_not_rendered':
        raise RuntimeError('refusing to reset metadata of an already started render batch')
    runtime=ROOT/'audit/2026-10-04/sequences';runtime.mkdir(parents=True,exist_ok=True)
    scenes={
        'moon_prinzvent':{'dem':'dem.img','albedo':'albedo.tiff','reference_image':'dom.img'},
        'southpole':{'dem':'dem.tif'},
    }
    resources={}
    for scene,roles in scenes.items():
        resources[scene]={}
        for role,name in roles.items():
            source=ROOT/'assets'/scene/name;target=destination/'maps'/scene/name
            copy_guarded(source,target,destination)
            info=runtime/f'{scene}_{name}.json'
            subprocess.run([str(PROJECT/'bin/resource_info'),str(source),str(info)],check=True)
            metadata=json.loads(info.read_text());metadata['dataset_file']=str(target.relative_to(destination));metadata['role']=role
            metadata['sha256']=checksum(source)
            resources[scene][role]=metadata
        georef=resources[scene]['dem'].copy()
        if not georef['origin_valid']:raise RuntimeError(f'{scene} DEM reference origin is NoData')
        budget.write_json(destination/'maps'/scene/'georeference.json',georef)
    sequences=[]
    for trajectory in sorted((ROOT/'assets').rglob('trajectory*.txt')):
        scene=trajectory.parent.name
        for device in ['davis346','prophesee']:
            config=PROJECT/'config'/f'{device}.yaml';cfg=yaml.safe_load(config.read_text());intr=cfg['intrinsic']
            name=f'{scene}_{trajectory.stem}_{device}';out=destination/'sequences'/name;out.mkdir(parents=True,exist_ok=True)
            copy_guarded(config,out/'source_config.yaml',destination)
            copy_guarded(trajectory,out/'source_trajectory.txt',destination)
            camera={'model':'pinhole','resolution':[intr['width'],intr['height']],
                    'K':[[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]],
                    'distortion_model':'plumb_bob','distortion_coefficients':[0,0,0,0,0],
                    'distortion_provenance':'exact coefficients of the simulated ideal pinhole; not measured hardware calibration',
                    'pixel_centers':'integer','image_axes':'x_right_y_down',
                    'pose_axes':'opencv_x_right_y_down_z_forward',
                    'T_blender_camera_from_opencv_camera':[[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],
                    'R_rectification':[[1,0,0],[0,1,0],[0,0,1]],'intrinsics_source':str(config)}
            budget.write_text(out/'calibration.yaml',yaml.safe_dump(camera,sort_keys=False))
            manifest={'sequence_id':name,'duration_seconds':60,'camera':device,'source_scene':scene,
                      'source_trajectory':str(trajectory),'source_config':str(config),
                      'motion_extension':'original-speed ping-pong with 0.5 s C2 phase reversal, no global slowing',
                      'map_reference':str(Path('../../maps')/scene/'georeference.json'),
                      'resources':{role:meta['dataset_file'] for role,meta in resources[scene].items()},
                      'physics_calibration':'generic uncalibrated circuit/optical parameters',
                      'status':'prepared_not_rendered','minimum_required_outputs':['events.h5','frames.h5','poses.csv','motion.csv','pose_geographic.csv']}
            existing=out/'sequence.json'
            if existing.exists() and json.loads(existing.read_text()).get('status')!='prepared_not_rendered':
                raise RuntimeError(f'refusing to reset a started sequence {name}')
            budget.write_json(out/'sequence.json',manifest);sequences.append(manifest)
    plan={'duration_seconds':60,'sequences':sequences,'resources':resources,
          'unused_resource':{'path':str(ROOT/'assets/starmap_2020_32k.exr'),'reason':'star environment omitted by user authorization'},
          'max_dataset_bytes':LIMIT,'stop_before_bytes':480_000_000_000,
          'size_limit_action':'stop all further generation before the next reserved write; retain completed data and report volume by type',
          'dataset_contents':'data, calibration, georeference, source provenance and completion metadata only',
          'status':'prepared_not_rendered'}
    budget.write_json(destination/'dataset.json',plan)
    print(json.dumps({'prepared_sequences':len(sequences),'logical_bytes':occupied(destination),'destination':str(destination)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='/home/temp/cbf_datasets/simulate');a=p.parse_args();prepare(Path(a.output))
