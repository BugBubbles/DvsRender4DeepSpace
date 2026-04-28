import argparse
import os

import bpy
import mathutils
import numpy as np


def flatten_matrix_row_major(mat):
    arr = np.array(mat, dtype=np.float32)
    return arr.reshape(-1)


def collect_mesh_data(scene, depsgraph):
    vertices = []
    normals = []
    indices = []

    for obj in scene.objects:
        if obj.type != "MESH":
            continue

        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue

        mesh.calc_loop_triangles()
        normal_matrix = eval_obj.matrix_world.to_3x3()

        for tri in mesh.loop_triangles:
            base = len(vertices)
            for loop_idx in tri.loops:
                loop = mesh.loops[loop_idx]
                vert = mesh.vertices[loop.vertex_index]
                world_v = eval_obj.matrix_world @ vert.co
                world_n = (normal_matrix @ loop.normal).normalized()
                vertices.append((world_v.x, world_v.y, world_v.z))
                normals.append((world_n.x, world_n.y, world_n.z))
            indices.append((base, base + 1, base + 2))

        eval_obj.to_mesh_clear()

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def collect_frames(scene, depsgraph, camera, light, start_frame, end_frame):
    scale = scene.render.resolution_percentage / 100.0
    width = int(scene.render.resolution_x * scale)
    height = int(scene.render.resolution_y * scale)
    fps = scene.render.fps / scene.render.fps_base

    frames = []
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        depsgraph = bpy.context.evaluated_depsgraph_get()
        cam_eval = camera.evaluated_get(depsgraph)
        light_eval = light.evaluated_get(depsgraph)

        cam_world = flatten_matrix_row_major(cam_eval.matrix_world)
        cam_origin = cam_eval.matrix_world.translation

        light_dir = (
            light_eval.matrix_world.to_3x3() @ mathutils.Vector((0.0, 0.0, 1.0))
        ).normalized()
        light_intensity = float(light_eval.data.energy)

        timestamp = frame / fps

        payload = np.concatenate(
            [
                np.array([timestamp], dtype=np.float32),
                np.array([cam_origin.x, cam_origin.y, cam_origin.z], dtype=np.float32),
                cam_world.astype(np.float32),
                np.array([light_dir.x, light_dir.y, light_dir.z], dtype=np.float32),
                np.array([light_intensity], dtype=np.float32),
            ]
        )
        assert payload.shape[0] == 24
        frames.append(payload)

    return width, height, fps, np.vstack(frames).astype(np.float32)


def write_metadata(scene, camera, path, width, height, fps, frame_count):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_eval = camera.evaluated_get(depsgraph)
    # 获取相机内参数矩阵
    focal_length = cam_eval.data.lens
    sensor_width = cam_eval.data.sensor_width

    fx = focal_length * (width / sensor_width)
    fy = focal_length * (height / sensor_width)
    cx = width / 2.0
    cy = height / 2.0
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"fps={fps}\n")
        f.write(f"frame_count={frame_count}\n")
        f.write(f"fx={fx}\n")
        f.write(f"fy={fy}\n")
        f.write(f"cx={cx}\n")
        f.write(f"cy={cy}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="export bundle directory")
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera = scene.camera
    if camera is None:
        raise RuntimeError("No camera in scene")
    light = next((obj for obj in scene.objects if obj.type == "LIGHT"), None)
    if light is None:
        raise RuntimeError("No light in scene")

    start_frame = args.start_frame if args.start_frame is not None else scene.frame_start
    end_frame = args.end_frame if args.end_frame is not None else scene.frame_end
    if end_frame < start_frame:
        raise RuntimeError("end_frame must be >= start_frame")

    os.makedirs(args.output_dir, exist_ok=True)

    vertices, normals, indices = collect_mesh_data(scene, depsgraph)
    width, height, fps, frames = collect_frames(
        scene, depsgraph, camera, light, start_frame, end_frame
    )

    vertices.astype(np.float32).tofile(os.path.join(args.output_dir, "mesh_vertices.bin"))
    normals.astype(np.float32).tofile(os.path.join(args.output_dir, "mesh_normals.bin"))
    indices.astype(np.uint32).tofile(os.path.join(args.output_dir, "mesh_indices.bin"))
    frames.astype(np.float32).tofile(os.path.join(args.output_dir, "frames.bin"))

    write_metadata(
        scene, camera,
        os.path.join(args.output_dir, "metadata.txt"),
        width=width,
        height=height,
        fps=fps,
        frame_count=frames.shape[0],
    )

    print(f"Export complete: {args.output_dir}")
    print(f"Vertices: {len(vertices)}, Triangles: {len(indices)}, Frames: {len(frames)}")


if __name__ == "__main__":
    import sys

    main()
