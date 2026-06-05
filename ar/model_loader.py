from pathlib import Path

import numpy as np


class OBJModel:
    def __init__(self, vertices, faces, face_colors=None):
        self.vertices = vertices
        self.faces = faces
        self.face_colors = face_colors or []


class ModelLoader:
    """Minimal OBJ loader with simple vertex-color and MTL diffuse-color support."""

    def __init__(self):
        self.cache = {}

    def load(self, path):
        if not path:
            return None

        model_path = Path(path)
        if not model_path.exists():
            return None
        key = str(model_path.resolve())
        if key in self.cache:
            return self.cache[key]

        vertices = []
        vertex_colors = []
        faces = []
        face_colors = []
        materials = {}
        current_material = None
        with model_path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "mtllib" and len(parts) >= 2:
                    for material_file in parts[1:]:
                        materials.update(self._load_mtl(model_path.parent / material_file))
                elif parts[0] == "usemtl" and len(parts) >= 2:
                    current_material = parts[1]
                elif parts[0] == "v" and len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    if len(parts) >= 7:
                        vertex_colors.append(self._parse_vertex_color(parts[4:7]))
                    else:
                        vertex_colors.append(None)
                elif parts[0] == "f" and len(parts) >= 4:
                    indices = []
                    for item in parts[1:]:
                        index_text = item.split("/")[0]
                        if not index_text:
                            continue
                        index = int(index_text)
                        if index < 0:
                            index = len(vertices) + index + 1
                        indices.append(index - 1)
                    for idx in range(1, len(indices) - 1):
                        face = [indices[0], indices[idx], indices[idx + 1]]
                        faces.append(face)
                        face_colors.append(
                            self._resolve_face_color(face, vertex_colors, materials, current_material)
                        )

        if not vertices or not faces:
            return None

        model = OBJModel(
            self._normalize_vertices(np.asarray(vertices, dtype=np.float32)),
            faces,
            face_colors=face_colors,
        )
        self.cache[key] = model
        return model

    def _load_mtl(self, path):
        materials = {}
        if not path.exists():
            return materials

        current_name = None
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "newmtl" and len(parts) >= 2:
                    current_name = parts[1]
                elif parts[0] == "Kd" and current_name and len(parts) >= 4:
                    materials[current_name] = self._parse_rgb_color(parts[1:4])
        return materials

    def _parse_vertex_color(self, values):
        return self._parse_rgb_color(values)

    def _parse_rgb_color(self, values):
        rgb = [float(value) for value in values[:3]]
        if max(rgb) <= 1.0:
            rgb = [value * 255.0 for value in rgb]
        rgb = [max(0, min(255, int(round(value)))) for value in rgb]
        return (rgb[2], rgb[1], rgb[0])

    def _resolve_face_color(self, face, vertex_colors, materials, current_material):
        material_color = materials.get(current_material)
        if material_color is not None:
            return material_color

        colors = [vertex_colors[index] for index in face if 0 <= index < len(vertex_colors) and vertex_colors[index] is not None]
        if not colors:
            return None

        return tuple(int(round(sum(color[channel] for color in colors) / len(colors))) for channel in range(3))

    def _normalize_vertices(self, vertices):
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        center = (minimum + maximum) * 0.5
        size = float(np.max(maximum - minimum))
        if size <= 1e-6:
            size = 1.0
        normalized = (vertices - center) / size
        normalized[:, 2] -= normalized[:, 2].min()
        return normalized.astype(np.float32)
