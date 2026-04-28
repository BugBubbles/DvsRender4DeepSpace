import argparse
import os
import sys

import bpy


CHANNEL_MAP = {
    ("location", 0): "loc_x",
    ("location", 1): "loc_y",
    ("location", 2): "loc_z",
    ("rotation_quaternion", 0): "quat_w",
    ("rotation_quaternion", 1): "quat_x",
    ("rotation_quaternion", 2): "quat_y",
    ("rotation_quaternion", 3): "quat_z",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise RuntimeError("No camera in scene")
    if cam.animation_data is None or cam.animation_data.action is None:
        raise RuntimeError("Camera has no animation action")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    action = cam.animation_data.action

    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write("# channel,frame,value,hl_frame,hl_value,hr_frame,hr_value\n")
        for fc in action.fcurves:
            key = (fc.data_path, fc.array_index)
            if key not in CHANNEL_MAP:
                continue
            channel = CHANNEL_MAP[key]
            for k in fc.keyframe_points:
                hl = k.handle_left
                hr = k.handle_right
                f.write(
                    f"{channel},{k.co.x},{k.co.y},"
                    f"{hl.x},{hl.y},{hr.x},{hr.y}\n"
                )

    print(f"Exported camera fcurves to {args.output_csv}")


if __name__ == "__main__":
    main()
