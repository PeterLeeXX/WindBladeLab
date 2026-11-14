"""Three rectangular blades, outward triangles, and fitted long-side surfaces."""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np


_DIRECTION_TOL = 1e-8


def _scalar(value, name, *, positive=False):
    try:
        if np.ndim(value) != 0 or isinstance(value, (bool, np.bool_)):
            raise ValueError
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite scalar") from None
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _array(value, name):
    try:
        result = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite numeric array") from None
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, eq=False)
class Turbine:
    """Dimensions in metres; base position and right-handed yaw/rotor angles."""

    blade_length: float
    blade_width: float
    tower_height: float
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    rotor_angle: float = 0.0

    def __post_init__(self):
        for name in ("blade_length", "blade_width", "tower_height", "yaw", "rotor_angle"):
            object.__setattr__(self, name, _scalar(
                getattr(self, name), name,
                positive=name in ("blade_length", "blade_width", "tower_height"),
            ))
        position = _array(self.position, "position")
        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")
        position.setflags(write=False)
        object.__setattr__(self, "position", position)


def _frame(turbine, blade):
    yaw = turbine.yaw % (2 * np.pi)
    angle = (turbine.rotor_angle % (2 * np.pi)) + blade * 2 * np.pi / 3
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    c, s = np.cos(angle), np.sin(angle)
    axial = rotation[:, 0]
    tangent = rotation @ np.array([0, c, s])
    axis = rotation @ np.array([0, -s, c])
    hub = turbine.position + np.array([0, 0, turbine.tower_height])
    return hub, axial, tangent, axis


def _triangle_data(triangles):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        edges = triangles[:, 1:] - triangles[:, :1]
        cross = np.cross(edges[:, 0], edges[:, 1])
        areas = np.linalg.norm(cross, axis=1)
        normals = cross / areas[:, None]
        centroids = triangles[:, 0] + edges.sum(axis=1) / 3
    if np.any(areas <= 0) or not np.all(np.isfinite(normals)) or not np.all(np.isfinite(centroids)):
        raise ValueError("triangles must be finite and nondegenerate")
    return centroids, normals


def _normal_errors(triangles):
    """Bound unit-normal error from rounded world vertices and edge conditioning."""
    eps = np.finfo(float).eps
    edges = triangles[:, 1:] - triangles[:, :1]
    lengths = np.linalg.norm(edges, axis=2)
    area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
    # Two world-coordinate additions in generation: allow 2 eps per coordinate.
    # Each edge subtracts two such vertices, plus its own subtraction roundoff.
    vertex_error = 2 * eps * np.linalg.norm(np.max(np.abs(triangles), axis=1), axis=1)
    edge_error = 2 * vertex_error[:, None] + eps * lengths
    # ||delta(e1 x e2)|| <= d1 L2 + d2 L1 + d1 d2,
    # plus cross-product arithmetic (products, subtraction and norm rounding).
    cross_error = (edge_error[:, 0] * lengths[:, 1]
                   + edge_error[:, 1] * lengths[:, 0]
                   + edge_error.prod(axis=1) + 8 * eps * lengths.prod(axis=1))
    if np.any(cross_error >= area) or not np.all(np.isfinite(cross_error)):
        raise ValueError("triangle normals are unresolved at coordinate precision")
    # Normalization changes direction by at most 2 ||delta cross|| / ||cross||.
    return 2 * cross_error / area


@dataclass(frozen=True, eq=False)
class Mesh:
    """Generated triangle vertices [N,3,3], physical blade IDs [N], and model."""

    triangles: np.ndarray
    blade_ids: np.ndarray
    turbine: Turbine

    def __post_init__(self):
        if not isinstance(self.turbine, Turbine):
            raise ValueError("turbine must be a Turbine")
        triangles = _array(self.triangles, "triangles")
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
            raise ValueError("triangles must have nonempty shape (N, 3, 3)")
        ids = _array(self.blade_ids, "blade_ids")
        if ids.shape != (len(triangles),) or np.any((ids < 0) | (ids > 2) | (ids != np.floor(ids))):
            raise ValueError("blade_ids must have shape (N,) and integer values 0, 1, 2")
        _triangle_data(triangles)
        ids = ids.astype(np.int64)
        triangles.setflags(write=False)
        ids.setflags(write=False)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "blade_ids", ids)


def generate_mesh(turbine, *, segments):
    """Generate four triangulated long sides and two caps per physical blade."""
    if not isinstance(turbine, Turbine):
        raise ValueError("turbine must be a Turbine")
    if isinstance(segments, (bool, np.bool_)) or not isinstance(segments, Integral) or segments <= 0:
        raise ValueError("segments must be a positive integer")
    triangles, ids = [], []

    def quad(vertices, blade):
        triangles.extend([vertices[[0, 1, 2]], vertices[[0, 2, 3]]])
        ids.extend([blade, blade])

    for blade in range(3):
        hub, axial, tangent, axis = _frame(turbine, blade)
        corners = np.array([
            -axial * turbine.blade_width / 6 - tangent * turbine.blade_width / 2,
            axial * turbine.blade_width / 6 - tangent * turbine.blade_width / 2,
            axial * turbine.blade_width / 6 + tangent * turbine.blade_width / 2,
            -axial * turbine.blade_width / 6 + tangent * turbine.blade_width / 2,
        ])
        rings = hub + corners + np.linspace(0, turbine.blade_length, segments + 1)[:, None, None] * axis
        for side in range(4):
            next_side = (side + 1) % 4
            for strip in range(segments):
                quad(np.array([rings[strip, side], rings[strip, next_side],
                               rings[strip + 1, next_side], rings[strip + 1, side]]), blade)
        quad(rings[0, [3, 2, 1, 0]], blade)
        quad(rings[-1], blade)
    return Mesh(np.array(triangles), np.array(ids), turbine)


@dataclass(frozen=True, eq=False)
class Surface:
    """Internal planner input: stable ID, root/tip endpoints [2,3], unit normal [3]."""

    id: tuple[int, int]
    endpoints: np.ndarray
    normal: np.ndarray


def _group_surfaces(mesh):
    """Return twelve Surfaces sorted by (blade, side), independent of triangle order.

    Side IDs in the blade frame: 0=-tangent, 1=+axial, 2=+tangent,
    3=-axial. Caps are excluded by their normals, not by triangle indices.
    """
    if not isinstance(mesh, Mesh):
        raise ValueError("mesh must be a Mesh")
    centroids, normals = _triangle_data(mesh.triangles)
    normal_errors = _normal_errors(mesh.triangles)
    turbine = mesh.turbine
    scale = max(turbine.blade_length, turbine.blade_width)
    length_tol = scale * 1e-8 + 32 * np.finfo(float).eps * np.max(np.abs(mesh.triangles))
    surfaces = []
    for blade in range(3):
        hub, axial, tangent, axis = _frame(turbine, blade)
        expected = np.array([-tangent, axial, tangent, -axial])
        widths = np.array([turbine.blade_width / 2, turbine.blade_width / 6] * 2)
        indices = np.flatnonzero(mesh.blade_ids == blade)
        # Sorting by geometry makes representative selection independent of storage order.
        indices = sorted(indices, key=lambda i: (*normals[i], *centroids[i]))
        groups = []
        for index in indices:
            normal, center = normals[index], centroids[index]
            if min(np.linalg.norm(normal - axis), np.linalg.norm(normal + axis)) <= (
                    _DIRECTION_TOL + normal_errors[index]):
                continue
            for group in groups:
                first = group[0]
                if (np.linalg.norm(normal - normals[first]) <= (
                        _DIRECTION_TOL + normal_errors[index] + normal_errors[first])
                        and abs((center - centroids[first]) @ normal) <= (
                            length_tol + np.linalg.norm(center - centroids[first])
                            * normal_errors[index])):
                    group.append(index)
                    break
            else:
                groups.append([index])
        if len(groups) != 4:
            raise ValueError(f"blade {blade} must have exactly four long-side groups")
        found = set()
        for group in groups:
            points = centroids[group]
            normal = normals[group].mean(axis=0)
            normal /= np.linalg.norm(normal)
            side = int(np.argmin(np.linalg.norm(expected - normal, axis=1)))
            if side in found or np.any(
                    np.linalg.norm(normals[group] - expected[side], axis=1)
                    > _DIRECTION_TOL + normal_errors[group]):
                raise ValueError(f"blade {blade} has invalid long-side normals")
            found.add(side)
            vertices = mesh.triangles[group] - hub
            # Check actual vertices against the physical plane without amplifying
            # the measured normal's angular uncertainty over the blade length.
            if np.any(np.abs(vertices @ expected[side] - widths[side]) > length_tol):
                raise ValueError(f"blade {blade} side {side} is not on its physical plane")
            longitudinal = (points - hub) @ axis
            # z is the primary key; longitudinal position breaks horizontal ties.
            z = points[:, 2]
            order = np.lexsort((longitudinal, z))
            if np.ptp(z) <= length_tol:
                order = np.argsort(longitudinal, kind="stable")
            points, longitudinal = points[order], longitudinal[order]
            if len(group) % 2:
                raise ValueError("long-side groups must contain paired strip triangles")
            strips = len(group) // 2
            strip_ids = np.floor(longitudinal / turbine.blade_length * strips).astype(int)
            if np.any((strip_ids < 0) | (strip_ids >= strips)):
                raise ValueError("triangle centroids lie outside physical blade length")
            centers, parameters = [], []
            for strip in dict.fromkeys(strip_ids.tolist()):
                selected = strip_ids == strip
                if np.count_nonzero(selected) != 2:
                    raise ValueError("each longitudinal strip must contain two triangles")
                bounds = vertices[order][selected] @ axis
                target = np.array([strip, strip + 1]) * turbine.blade_length / strips
                if not np.allclose([bounds.min(), bounds.max()], target, rtol=0, atol=length_tol):
                    raise ValueError("strip triangles must span their physical longitudinal interval")
                expected_parameters = (strip + np.array([1 / 3, 2 / 3])) * turbine.blade_length / strips
                if not np.allclose(np.sort(longitudinal[selected]), expected_parameters,
                                   rtol=0, atol=length_tol):
                    raise ValueError("strip triangles must form a complementary pair")
                centers.append(points[selected].mean(axis=0) - hub)
                parameters.append(longitudinal[selected].mean())
            if len(centers) != strips:
                raise ValueError("long-side group has missing longitudinal strips")
            centers, parameters = np.array(centers), np.array(parameters)
            # Fit residuals around the known longitudinal axis. With one strip the
            # minimum-norm residual slope is zero, retaining the physical axis.
            u = parameters / turbine.blade_length - 0.5
            design = np.column_stack([np.ones(strips), u])
            if strips == 1:
                design[:, 1] = 0
            residuals = centers - parameters[:, None] * axis
            coefficients = np.linalg.lstsq(design, residuals, rcond=None)[0]
            endpoints = (hub + np.array([[1, -0.5], [1, 0.5]]) @ coefficients
                         + np.array([[0], [turbine.blade_length]]) * axis)
            if not np.all(np.isfinite(endpoints)):
                raise ValueError("surface endpoints must be finite")
            endpoints.setflags(write=False)
            normal.setflags(write=False)
            surfaces.append(Surface((blade, side), endpoints, normal))
    return tuple(sorted(surfaces, key=lambda surface: surface.id))
