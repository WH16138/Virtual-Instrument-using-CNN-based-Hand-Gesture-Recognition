from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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
        self.mesh_futures = {}
        self.mesh_lock = Lock()
        self.preload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model-preload")
        self.renderer = None
        self.renderer_size = None
        self.texture_renderer = None
        self.texture_renderer_size = None
        self.texture_cache = {}
        self.last_overlay = None
        self.reported_failures = set()

    def preload_models(self, model_paths):
        if not self.available:
            return
        for model_path in model_paths:
            path = Path(model_path) if model_path else None
            if path is None or path.suffix.lower() not in self.SUPPORTED_SUFFIXES or not path.exists():
                continue
            key = str(path.resolve())
            with self.mesh_lock:
                if key in self.mesh_cache or key in self.mesh_futures:
                    continue
                self.mesh_futures[key] = self.preload_executor.submit(self._load_meshes_sync, path)

    def prepare_viewport(self, width, height):
        if not self.available:
            return False
        return self._get_renderer(width, height) is not None

    def close(self):
        self.preload_executor.shutdown(wait=False, cancel_futures=True)
        if self.renderer is not None:
            try:
                self.renderer.delete()
            except Exception:
                pass
            self.renderer = None
            self.renderer_size = None
        if self.texture_renderer is not None:
            try:
                self.texture_renderer.delete()
            except Exception:
                pass
            self.texture_renderer = None
            self.texture_renderer_size = None

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

        meshes = self._load_meshes(path, allow_pending=False)
        if not meshes:
            return False

        height, width = frame.shape[:2]
        renderer = self._get_renderer(width, height)
        if renderer is None:
            return False

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.35, 0.35, 0.35, 1.0])
        model_pose = self._model_pose(board_pos, size, height_offset, yaw_degrees, pose.get("z_sign", 1.0))
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
        camera_pose = self._camera_pose_from_solvepnp(pose["rvec"], pose["tvec"])
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.6), pose=camera_pose)

        try:
            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
        except Exception:
            return False

        if rgba.shape[2] < 4 or np.max(rgba[:, :, 3]) <= 0:
            return False
        self._alpha_blend(frame, rgba, alpha)
        return True

    def render_models(self, frame, pose, model_specs, render_scale=1.0):
        """Render several board-space models in one pyrender pass.

        render_scale renders into a smaller offscreen buffer and upscales the
        RGBA result before compositing. This is useful for animated monster
        models, where full camera resolution rendering is the main FPS bottleneck.
        """
        statuses = [False] * len(model_specs or [])
        if not self.available or pose is None or not model_specs:
            return statuses

        height, width = frame.shape[:2]
        render_scale = float(np.clip(render_scale, 0.35, 1.0))
        render_width = max(1, int(round(width * render_scale)))
        render_height = max(1, int(round(height * render_scale)))
        renderer = self._get_renderer(render_width, render_height)
        if renderer is None:
            return statuses

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.35, 0.35, 0.35, 1.0])
        max_alpha = 0.0
        for index, spec in enumerate(model_specs):
            model_path = spec.get("model_path")
            if not model_path:
                continue

            path = Path(model_path)
            if path.suffix.lower() not in self.SUPPORTED_SUFFIXES or not path.exists():
                continue

            meshes = self._load_meshes(path, allow_pending=False)
            if not meshes:
                continue

            fit_size = spec.get("fit_size")
            if fit_size is not None:
                model_pose = self._model_pose_fit(
                    meshes,
                    spec.get("board_pos", (0.0, 0.0)),
                    fit_size,
                    float(spec.get("height_offset", 0.0)),
                    float(spec.get("yaw_degrees", 0.0)),
                    pose.get("z_sign", 1.0),
                )
            else:
                model_pose = self._model_pose(
                    spec.get("board_pos", (0.0, 0.0)),
                    float(spec.get("size", 1.0)),
                    float(spec.get("height_offset", 0.0)),
                    float(spec.get("yaw_degrees", 0.0)),
                    pose.get("z_sign", 1.0),
                )
            for mesh in meshes:
                scene.add(mesh, pose=model_pose)
            statuses[index] = True
            max_alpha = max(max_alpha, float(spec.get("alpha", 1.0)))

        if not any(statuses):
            return statuses

        camera_pose = self._camera_pose_from_solvepnp(pose["rvec"], pose["tvec"])
        camera = pyrender.IntrinsicsCamera(
            fx=float(pose["camera_matrix"][0, 0]) * render_scale,
            fy=float(pose["camera_matrix"][1, 1]) * render_scale,
            cx=float(pose["camera_matrix"][0, 2]) * render_scale,
            cy=float(pose["camera_matrix"][1, 2]) * render_scale,
            znear=1.0,
            zfar=10000.0,
        )
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.6), pose=camera_pose)

        try:
            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
        except Exception:
            return [False] * len(statuses)

        if rgba.shape[2] < 4 or np.max(rgba[:, :, 3]) <= 0:
            return [False] * len(statuses)

        blend_alpha = max_alpha if max_alpha > 0.0 else 1.0
        self.last_overlay = {
            "rgba": rgba.copy(),
            "render_scale": render_scale,
            "frame_shape": (height, width),
            "alpha": blend_alpha,
        }
        if render_width != width or render_height != height:
            self._alpha_blend_scaled(frame, rgba, render_scale, blend_alpha)
        else:
            self._alpha_blend(frame, rgba, blend_alpha)
        return statuses

    def blend_last_overlay(self, frame):
        if not self.last_overlay:
            return False
        height, width = frame.shape[:2]
        if self.last_overlay.get("frame_shape") != (height, width):
            return False
        rgba = self.last_overlay.get("rgba")
        if rgba is None:
            return False
        render_scale = float(self.last_overlay.get("render_scale", 1.0))
        alpha = float(self.last_overlay.get("alpha", 1.0))
        if render_scale < 0.999:
            self._alpha_blend_scaled(frame, rgba, render_scale, alpha)
        else:
            self._alpha_blend(frame, rgba, alpha)
        return True

    def render_topdown_texture(self, model_path, texture_size=512, allow_pending=False):
        """Render a cached top-down RGBA sprite for flat ground assets."""
        if not self.available or not model_path:
            return None

        path = Path(model_path)
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES or not path.exists():
            return None

        texture_size = int(max(128, min(1024, texture_size)))
        key = (str(path.resolve()), texture_size)
        cached = self.texture_cache.get(key)
        if cached is not None:
            return cached

        meshes = self._load_meshes(path, allow_pending=allow_pending)
        if not meshes:
            return None

        renderer = self._get_texture_renderer(texture_size)
        if renderer is None:
            return None

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.52, 0.52, 0.52, 1.0])
        for mesh in meshes:
            scene.add(mesh, pose=np.eye(4, dtype=np.float64))

        camera_pose = np.eye(4, dtype=np.float64)
        camera_pose[2, 3] = 2.2
        camera = pyrender.OrthographicCamera(xmag=0.68, ymag=0.68, znear=0.01, zfar=10.0)
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.8), pose=camera_pose)

        try:
            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES)
        except Exception:
            return None

        if rgba.shape[2] < 4 or np.max(rgba[:, :, 3]) <= 0:
            return None

        self.texture_cache[key] = rgba
        return rgba

    def _get_renderer(self, width, height):
        size = (int(width), int(height))
        if self.renderer is not None and self.renderer_size == size:
            return self.renderer
        try:
            if self.renderer is not None:
                self.renderer.delete()
            self.renderer = pyrender.OffscreenRenderer(viewport_width=size[0], viewport_height=size[1])
            self.renderer_size = size
        except Exception:
            self.renderer = None
            self.renderer_size = None
        return self.renderer

    def _get_texture_renderer(self, texture_size):
        size = int(texture_size)
        if self.texture_renderer is not None and self.texture_renderer_size == size:
            return self.texture_renderer
        try:
            if self.texture_renderer is not None:
                self.texture_renderer.delete()
            self.texture_renderer = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size)
            self.texture_renderer_size = size
        except Exception:
            self.texture_renderer = None
            self.texture_renderer_size = None
        return self.texture_renderer

    def _load_meshes(self, path, allow_pending=True):
        key = str(path.resolve())
        with self.mesh_lock:
            if key in self.mesh_cache:
                return self.mesh_cache[key]
            future = self.mesh_futures.get(key)

        if future is not None:
            if not future.done() and not allow_pending:
                return None
            try:
                meshes = future.result()
            except Exception:
                meshes = None
            with self.mesh_lock:
                self.mesh_futures.pop(key, None)
                if meshes:
                    self.mesh_cache[key] = meshes
            return meshes

        meshes = self._load_meshes_sync(path)
        if meshes:
            with self.mesh_lock:
                self.mesh_cache[key] = meshes
        return meshes

    def _load_meshes_sync(self, path):
        key = str(path.resolve())
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

    def _mesh_position_bounds(self, meshes):
        positions = []
        for mesh in meshes:
            for primitive in getattr(mesh, "primitives", []):
                primitive_positions = getattr(primitive, "positions", None)
                if primitive_positions is not None and len(primitive_positions) > 0:
                    positions.append(np.asarray(primitive_positions, dtype=np.float64))
        if not positions:
            return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
        stacked = np.vstack(positions)
        return np.min(stacked, axis=0), np.max(stacked, axis=0)

    def _model_pose_fit(self, meshes, board_pos, fit_size, height_offset, yaw_degrees=0.0, z_sign=1.0):
        target_width, target_height = fit_size
        bounds_min, bounds_max = self._mesh_position_bounds(meshes)
        extent = np.maximum(bounds_max - bounds_min, 1e-6)
        sx = float(target_width) / float(extent[0])
        sy = float(target_height) / float(extent[1])
        sz = max(sx, sy)
        z_sign = 1.0 if float(z_sign) >= 0.0 else -1.0

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
        scale_matrix = np.diag([sx, sy, sz * z_sign]).astype(np.float64)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation_z @ scale_matrix
        pose[0, 3] = float(board_pos[0])
        pose[1, 3] = float(board_pos[1])
        pose[2, 3] = float(height_offset) * z_sign
        return pose

    def _model_pose(self, board_pos, size, height_offset, yaw_degrees=0.0, z_sign=1.0):
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
        z_sign = 1.0 if float(z_sign) >= 0.0 else -1.0
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

    def _alpha_blend_scaled(self, frame, rgba, render_scale, global_alpha):
        bbox = self._alpha_bbox(rgba)
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        full_x1 = max(0, int(np.floor(x1 / render_scale)))
        full_y1 = max(0, int(np.floor(y1 / render_scale)))
        full_x2 = min(frame.shape[1], int(np.ceil(x2 / render_scale)))
        full_y2 = min(frame.shape[0], int(np.ceil(y2 / render_scale)))
        if full_x2 <= full_x1 or full_y2 <= full_y1:
            return

        crop = rgba[y1:y2, x1:x2]
        resized = cv2.resize(
            crop,
            (full_x2 - full_x1, full_y2 - full_y1),
            interpolation=cv2.INTER_LINEAR,
        )
        self._alpha_blend(frame[full_y1:full_y2, full_x1:full_x2], resized, global_alpha)

    def _alpha_blend(self, frame_roi, rgba, global_alpha):
        if rgba.shape[2] < 4 or frame_roi.size == 0:
            return

        bbox = self._alpha_bbox(rgba)
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        rgba_roi = rgba[y1:y2, x1:x2]
        target = frame_roi[y1:y2, x1:x2]

        rgb = rgba_roi[:, :, :3].astype(np.float32)
        alpha = (rgba_roi[:, :, 3:4].astype(np.float32) / 255.0) * float(global_alpha)
        if np.max(alpha) <= 0.0:
            return

        bgr = rgb[:, :, ::-1]
        blended = bgr * alpha + target.astype(np.float32) * (1.0 - alpha)
        np.copyto(target, blended.astype(np.uint8))

    def _alpha_bbox(self, rgba):
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            return None
        ys, xs = np.where(rgba[:, :, 3] > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        pad = 2
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(rgba.shape[1], int(xs.max()) + pad + 1)
        y2 = min(rgba.shape[0], int(ys.max()) + pad + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2
