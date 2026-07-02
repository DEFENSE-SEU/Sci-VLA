import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def _rounded_vector(values, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _rounded_float(value, digits: int = 4) -> float:
    return round(float(value), digits)


def _as_vec3(values) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        return None
    return arr


def _norm(vec: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        return vec
    return vec / length


def camera_summary_from_pose(
    *,
    view: str,
    camera_name: str,
    camera_pos,
    camera_xmat,
    fovy: float,
    image_shape,
) -> dict:
    height = int(image_shape[0])
    width = int(image_shape[1])
    fovy_rad = math.radians(float(fovy))
    fy = (0.5 * height) / math.tan(0.5 * fovy_rad)
    fx = fy
    rotation = np.asarray(camera_xmat, dtype=np.float64).reshape(3, 3)

    return {
        "view": str(view),
        "camera_name": str(camera_name),
        "resolution": {"width": width, "height": height},
        "intrinsics": {
            "fx": round(float(fx), 4),
            "fy": round(float(fy), 4),
            "cx": round((width - 1) / 2.0, 4),
            "cy": round((height - 1) / 2.0, 4),
            "fovy_deg": round(float(fovy), 4),
        },
        "extrinsics": {
            "position_world": _rounded_vector(camera_pos),
            "image_right_world": _rounded_vector(rotation[:, 0]),
            "image_up_world": _rounded_vector(rotation[:, 1]),
            "optical_axis_world": _rounded_vector(-rotation[:, 2]),
        },
    }


def _axis_hint(axis: list[float]) -> str:
    labels = ["x", "y", "z"]
    if axis is None:
        return "unknown"
    arr = np.asarray(axis, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        return "unknown"
    index = int(np.argmax(np.abs(arr)))
    sign = "+" if arr[index] >= 0 else "-"
    return f"{sign}{labels[index]}"


def _camera_line(camera: dict) -> str:
    extrinsics = camera.get("extrinsics", {})
    right = extrinsics.get("image_right_world")
    up = extrinsics.get("image_up_world")
    optical = extrinsics.get("optical_axis_world")
    return (
        f"- {camera.get('view')}/{camera.get('camera_name')}: "
        f"image_right ~= {_axis_hint(right)}, image_up ~= {_axis_hint(up)}, "
        f"optical_axis ~= {_axis_hint(optical)}, "
        f"camera_pos={extrinsics.get('position_world')}"
    )


def format_calibration_for_llm(payload: dict | None) -> str:
    if not isinstance(payload, dict):
        return ""
    cameras = payload.get("cameras") or []
    lines = [
        "CAMERA CALIBRATION AND GEOMETRY",
        "World frame: MuJoCo coordinates. Use these calibrated axes and positions to interpret front/side images.",
    ]
    for camera in cameras:
        lines.append(_camera_line(camera))

    ee = payload.get("end_effector")
    if isinstance(ee, dict):
        lines.append(
            f"- end_effector/{ee.get('site_name')}: position_world={ee.get('position_world')}"
        )
    lines.append(
        "Use the images for visual evidence, but use this calibration text for spatial direction, depth, and clearance reasoning."
    )
    return "\n".join(lines)


def _bbox_center(bbox) -> tuple[float, float] | None:
    if bbox is None:
        return None
    values = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if values.size != 4:
        return None
    x1, y1, x2, y2 = values.tolist()
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _camera_ray_from_pixel(camera: dict, pixel: tuple[float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    intrinsics = camera.get("intrinsics", {})
    extrinsics = camera.get("extrinsics", {})
    origin = _as_vec3(extrinsics.get("position_world"))
    right = _as_vec3(extrinsics.get("image_right_world"))
    up = _as_vec3(extrinsics.get("image_up_world"))
    optical = _as_vec3(extrinsics.get("optical_axis_world"))
    if origin is None or right is None or up is None or optical is None:
        return None
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except Exception:
        return None
    if abs(fx) <= 1e-9 or abs(fy) <= 1e-9:
        return None
    u, v = pixel
    direction = _norm(optical + ((u - cx) / fx) * right + ((cy - v) / fy) * up)
    return origin, direction


def _closest_point_between_rays(
    ray_a: tuple[np.ndarray, np.ndarray],
    ray_b: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, float]:
    origin_a, direction_a = ray_a
    origin_b, direction_b = ray_b
    matrix = np.stack([direction_a, -direction_b], axis=1)
    rhs = origin_b - origin_a
    params, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    point_a = origin_a + params[0] * direction_a
    point_b = origin_b + params[1] * direction_b
    return (point_a + point_b) / 2.0, float(np.linalg.norm(point_a - point_b))


def _distance_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def _camera_by_view(calibration_payload: dict) -> dict[str, dict]:
    return {
        str(camera.get("view")): camera
        for camera in calibration_payload.get("cameras", [])
        if isinstance(camera, dict)
    }


def estimate_obstacles_from_vlm_output(
    vlm_output: dict | None,
    calibration_payload: dict,
    *,
    current_gripper_position=None,
    target_gripper_position=None,
    path_clearance_threshold_m: float = 0.08,
) -> dict:
    obstacles_raw = []
    if isinstance(vlm_output, dict) and isinstance(vlm_output.get("obstacles"), list):
        obstacles_raw = vlm_output["obstacles"]
    cameras = _camera_by_view(calibration_payload)
    current = _as_vec3(current_gripper_position)
    if current is None:
        current = _as_vec3((calibration_payload.get("end_effector") or {}).get("position_world"))
    target = _as_vec3(target_gripper_position)
    if target is None:
        target = _as_vec3((calibration_payload.get("target_end_effector") or {}).get("position_world"))

    obstacles = []
    for index, item in enumerate(obstacles_raw):
        if not isinstance(item, dict):
            continue
        front_center = _bbox_center(item.get("front_bbox"))
        side_center = _bbox_center(item.get("side_bbox"))
        rays = []
        if front_center is not None and "front" in cameras:
            ray = _camera_ray_from_pixel(cameras["front"], front_center)
            if ray is not None:
                rays.append(ray)
        if side_center is not None and "side" in cameras:
            ray = _camera_ray_from_pixel(cameras["side"], side_center)
            if ray is not None:
                rays.append(ray)
        center = None
        triangulation_error = None
        if len(rays) >= 2:
            center, triangulation_error = _closest_point_between_rays(rays[0], rays[1])

        obstacle = {
            "name": str(item.get("name") or f"obstacle_{index}"),
            "front_bbox": item.get("front_bbox"),
            "side_bbox": item.get("side_bbox"),
            "risk_reason": str(item.get("risk_reason") or ""),
            "confidence": item.get("confidence"),
        }
        if center is not None:
            obstacle["center_world"] = _rounded_vector(center)
            obstacle["triangulation_error_m"] = _rounded_float(triangulation_error or 0.0)
            if current is not None:
                obstacle["distance_to_current_gripper_m"] = _rounded_float(np.linalg.norm(center - current))
            if current is not None and target is not None:
                obstacle["distance_to_path_m"] = _rounded_float(_distance_to_segment(center, current, target))
        else:
            obstacle["center_world"] = None
            obstacle["triangulation_error_m"] = None
        obstacles.append(obstacle)

    nearest_obstacles = sorted(
        [obs for obs in obstacles if obs.get("distance_to_current_gripper_m") is not None],
        key=lambda obs: obs["distance_to_current_gripper_m"],
    )[:5]
    path_obstacles = sorted(
        [
            obs
            for obs in obstacles
            if obs.get("distance_to_path_m") is not None
            and obs["distance_to_path_m"] <= float(path_clearance_threshold_m)
        ],
        key=lambda obs: obs["distance_to_path_m"],
    )[:5]

    target_relation = None
    if current is not None and target is not None:
        delta = target - current
        target_relation = {
            "delta_world_m": _rounded_vector(delta),
            "distance_m": _rounded_float(np.linalg.norm(delta)),
        }

    return {
        "source": "vlm_bbox_plus_calibration",
        "current_gripper_position_world": _rounded_vector(current) if current is not None else None,
        "target_gripper_position_world": _rounded_vector(target) if target is not None else None,
        "target_relative_to_current": target_relation,
        "obstacles": obstacles,
        "nearest_obstacles": nearest_obstacles,
        "path_obstacles": path_obstacles,
        "path_clearance_threshold_m": _rounded_float(path_clearance_threshold_m),
        "notes": [
            "Obstacle positions are estimated from VLM front/side bounding boxes and calibrated camera rays.",
            "No simulator geom metadata is used.",
        ],
    }


def format_spatial_context_for_llm(context: dict | None) -> str:
    if not isinstance(context, dict):
        return ""
    lines = ["SPATIAL CONTEXT FROM VLM+CALIBRATION TOOL"]
    relation = context.get("target_relative_to_current")
    if isinstance(relation, dict):
        lines.append(
            f"- target delta: {relation.get('delta_world_m')} m, "
            f"distance={relation.get('distance_m')} m"
        )
    elif context.get("target_gripper_position_world") is None:
        lines.append("- target delta: unavailable; target gripper world pose was not provided.")

    nearest = context.get("nearest_obstacles") or []
    lines.append("- nearest obstacles:")
    if nearest:
        for obstacle in nearest[:5]:
            lines.append(
                f"  - {obstacle.get('name')}: center={obstacle.get('center_world')}, "
                f"distance_to_gripper={obstacle.get('distance_to_current_gripper_m')} m, "
                f"reason={obstacle.get('risk_reason')}"
            )
    else:
        lines.append("  - none estimated")

    path_obstacles = context.get("path_obstacles") or []
    lines.append("- path obstacles:")
    if path_obstacles:
        for obstacle in path_obstacles[:5]:
            lines.append(
                f"  - {obstacle.get('name')}: distance_to_path={obstacle.get('distance_to_path_m')} m, "
                f"reason={obstacle.get('risk_reason')}"
            )
    else:
        lines.append("  - none estimated or target pose unavailable")
    lines.append("Use this as tool output; use image evidence to reject false positives.")
    return "\n".join(lines)


def _draw_annotation(image: np.ndarray, camera: dict) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return np.asarray(image, dtype=np.uint8)

    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    draw = ImageDraw.Draw(pil_image)
    width, _height = pil_image.size
    extrinsics = camera.get("extrinsics", {})
    lines = [
        f"{camera.get('view')}: {camera.get('camera_name')}",
        f"right ~= {_axis_hint(extrinsics.get('image_right_world'))}",
        f"up ~= {_axis_hint(extrinsics.get('image_up_world'))}",
        f"depth ~= {_axis_hint(extrinsics.get('optical_axis_world'))}",
    ]
    pad = 6
    line_height = 13
    box_width = min(width - 2 * pad, 220)
    box_height = pad * 2 + line_height * len(lines)
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((pad, pad + idx * line_height), line, fill=(255, 255, 255))
    return np.asarray(pil_image, dtype=np.uint8)


def write_calibration_assets(
    payload: dict,
    images: dict[str, np.ndarray],
    *,
    output_dir: str | Path = "logs",
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "transition_calibration.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cameras_by_view = {
        str(camera.get("view")): camera
        for camera in payload.get("cameras", [])
        if isinstance(camera, dict)
    }
    annotated_paths: dict[str, str] = {}
    for view, image in images.items():
        if image is None:
            continue
        camera = cameras_by_view.get(view, {"view": view, "camera_name": view, "extrinsics": {}})
        annotated = _draw_annotation(np.asarray(image, dtype=np.uint8), camera)
        image_path = output_path / f"current_{view}_calibrated.png"
        imageio.imwrite(image_path, annotated)
        annotated_paths[view] = str(image_path)

    return {
        "json_path": str(json_path),
        "annotated_image_paths": annotated_paths,
    }


def _mujoco_id(mujoco_module, model, obj_type, name: str) -> int:
    try:
        return int(mujoco_module.mj_name2id(model, obj_type, name))
    except Exception:
        return -1


def build_transition_calibration_payload(
    *,
    model,
    data,
    front_image,
    side_image,
    front_camera_name: str = "table_cam_front",
    side_camera_name: str = "table_cam_left",
    ee_site_name: str = "/ur:2f85:pinch",
) -> dict:
    import mujoco

    cameras = []
    for view, camera_name, image in (
        ("front", front_camera_name, front_image),
        ("side", side_camera_name, side_image),
    ):
        if image is None:
            continue
        camera_id = _mujoco_id(mujoco, model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            continue
        fovy = float(np.asarray(model.cam_fovy)[camera_id])
        cameras.append(
            camera_summary_from_pose(
                view=view,
                camera_name=camera_name,
                camera_pos=np.asarray(data.cam_xpos)[camera_id],
                camera_xmat=np.asarray(data.cam_xmat)[camera_id].reshape(3, 3),
                fovy=fovy,
                image_shape=np.asarray(image).shape,
            )
        )

    payload: dict[str, Any] = {
        "frame_convention": "MuJoCo world coordinates; camera optical axis uses the calibrated world direction from each fixed camera.",
        "cameras": cameras,
    }

    site_id = _mujoco_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if site_id >= 0:
        payload["end_effector"] = {
            "site_name": ee_site_name,
            "position_world": _rounded_vector(np.asarray(data.site_xpos)[site_id]),
        }
    return payload
