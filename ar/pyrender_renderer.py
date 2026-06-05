from pathlib import Path

import cv2
import numpy as np


try:
    import pyrender
    import trimesh
except ImportError:
    pyrender = None
    trimesh = None


class PyrenderModelRenderer:
    """Render textured GLB/GLTF/OBJ assets into the OpenCV camera frame."""

    SUPPORTED_SUFFIXES = {".glb", ".gltf", ".obj"}

    def __init__(self):
        self.available = pyrender is not None and trimesh is not None
        self.mesh_cache = {}
        self.renderer = None
        self.renderer_size = None
        self.reported_failures = set()

    def render_model(
        self,
        frame,
        model_path,
        pose,
        board_pos,
        size,
        height_offset=0.0,
        yaw_degrees=0.0,
        alpha=1.0,
    ):
        if not self.available or pose is None or not model_path:
            return False

        path = Path(model_path)
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES or not path.exists():
            return False

        meshes = self._load_meshes(path)
        if not meshes:
            return False

        height, width = frame.shape[:2]
        renderer = self._get_renderer(width, height)
        if renderer is None:
            return False

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.35, 0.35, 0.35, 1.0])
        model_pose = self._model_pose(board_pos, size, height_offset, pose.get("z_sign", 1.0), yaw_degrees)
        for mesh in meshes:
            scene.add(mesh, pose=model_pose)

        camera = pyrender.IntrinsicsCamera(
            fx=float(pose["camera_matrix"][0, 0]),
            fy=float(pose["camera_matrix"][1, 1]),
            cx=float(pose["camera_matrix"][0, 2]),
            cy=float(pose["camera_matrix"][1, 2]),
            znear=1.0,
            zfar=10000.0,
        )
        scene.add(camera, pose=self._camera_pose_from_solvepnp(pose["rvec"], pose["tvec"]))

        light_pose = self._camera_pose_from_solvepnp(pose["rvec"], pose["tvec"])
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.6), pose=light_pose)

        try:
            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
        except Exception:
            return False

        if rgba.shape[2] < 4 or np.max(rgba[:, :, 3]) <= 0:
            return False
        self._alpha_blend(frame, rgba, alpha)
        return True

    def _get_renderer(self, width, height):
        size = (width, height)
        if self.renderer is not None and self.renderer_size == size:
            return self.renderer
        try:
            if self.renderer is not None:
                self.renderer.delete()
            self.renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
            self.renderer_size = size
        except Exception:
            self.renderer = None
            self.renderer_size = None
        return self.renderer

    def _load_meshes(self, path):
        key = str(path.resolve())
        if key in self.mesh_cache:
            return self.mesh_cache[key]

        try:
            loaded = trimesh.load(str(path), force="scene")
        except Exception:
            self._report_once(f"load:{key}", f"Failed to load model: {path}")
            return None

        if isinstance(loaded, trimesh.Trimesh):
            scene = trimesh.Scene(loaded)
        else:
            scene = loaded

        normalized_meshes = self._normalize_scene_meshes(scene, source_suffix=path.suffix.lower())
        if not normalized_meshes:
            self._report_once(f"empty:{key}", f"Model has no renderable mesh: {path}")
            return None

        pyrender_meshes = []
        for mesh in normalized_meshes:
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue
            try:
                pyrender_meshes.append(pyrender.Mesh.from_trimesh(mesh, smooth=True))
            except Exception:
                continue

        if not pyrender_meshes:
            self._report_once(f"convert:{key}", f"Failed to convert model for pyrender: {path}")
            return None

        self.mesh_cache[key] = pyrender_meshes
        return pyrender_meshes

    def _normalize_scene_meshes(self, scene, source_suffix=""):
        axis_transform = self._asset_axis_transform(source_suffix)
        source_meshes = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph.get(node_name)
            geometry = scene.geometry.get(geometry_name)
            if geometry is None:
                continue
            mesh = geometry.copy()
            mesh.apply_transform(transform)
            if axis_transform is not None:
                mesh.apply_transform(axis_transform)
            source_meshes.append(mesh)

        if not source_meshes:
            return []

        bounds_min = np.min([mesh.bounds[0] for mesh in source_meshes], axis=0)
        bounds_max = np.max([mesh.bounds[1] for mesh in source_meshes], axis=0)
        center = (bounds_min + bounds_max) * 0.5
        extent = float(np.max(bounds_max - bounds_min))
        if extent <= 1e-6:
            extent = 1.0

        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = -center

        normalized_meshes = []
        min_z = None
        for mesh in source_meshes:
            normalized = mesh.copy()
            normalized.apply_transform(transform)
            normalized.apply_scale(1.0 / extent)
            normalized_meshes.append(normalized)
            mesh_min_z = float(normalized.bounds[0, 2])
            min_z = mesh_min_z if min_z is None else min(min_z, mesh_min_z)

        if min_z is None:
            return []

        lift = np.eye(4, dtype=np.float64)
        lift[2, 3] = -min_z
        for mesh in normalized_meshes:
            mesh.apply_transform(lift)
        return normalized_meshes

    def _asset_axis_transform(self, source_suffix):
        if source_suffix not in {".glb", ".gltf"}:
            return None

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return transform

    def _report_once(self, key, message):
        if key in self.reported_failures:
            return
        self.reported_failures.add(key)
        print(f"[PyrenderModelRenderer] {message}")

    def _model_pose(self, board_pos, size, height_offset, z_sign, yaw_degrees=0.0):
        yaw = np.deg2rad(float(yaw_degrees))
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))

        rotation_z = np.asarray(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        scale_matrix = np.diag([size, size, size * z_sign]).astype(np.float64)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation_z @ scale_matrix
        pose[0, 3] = float(board_pos[0])
        pose[1, 3] = float(board_pos[1])
        pose[2, 3] = float(height_offset) * z_sign
        return pose

    def _camera_pose_from_solvepnp(self, rvec, tvec):
        rotation, _ = cv2.Rodrigues(rvec)
        world_to_camera_cv = np.eye(4, dtype=np.float64)
        world_to_camera_cv[:3, :3] = rotation
        world_to_camera_cv[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        world_to_camera_gl = cv_to_gl @ world_to_camera_cv
        return np.linalg.inv(world_to_camera_gl)

    def _alpha_blend(self, frame, rgba, global_alpha):
        if rgba.shape[2] < 4:
            return

        rgb = rgba[:, :, :3].astype(np.float32)
        alpha = (rgba[:, :, 3:4].astype(np.float32) / 255.0) * float(global_alpha)
        if np.max(alpha) <= 0.0:
            return

        bgr = rgb[:, :, ::-1]
        blended = bgr * alpha + frame.astype(np.float32) * (1.0 - alpha)
        np.copyto(frame, blended.astype(np.uint8))
