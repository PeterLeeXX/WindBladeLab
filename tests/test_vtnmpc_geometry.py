"""Geometry contracts, independently checked in physical blade frames."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from planning.vtnmpc import Mesh, Turbine, generate_mesh
from planning.vtnmpc.geometry import _group_surfaces


def frame(turbine, blade):
    y, a = turbine.yaw, turbine.rotor_angle + blade * 2 * np.pi / 3
    yaw = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    rotor = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a), np.cos(a)]])
    return yaw @ rotor


@pytest.fixture
def turbine():
    return Turbine(50, 3, 120)


@pytest.mark.parametrize("segments", [1, 2, 7, 30])
@pytest.mark.parametrize("angle", [0, np.pi / 2, 0.371])
def test_geometry_dimensions_planes_and_centerlines(turbine, segments, angle):
    turbine = replace(turbine, rotor_angle=angle, yaw=0.23, position=[4, -7, 2])
    mesh = generate_mesh(turbine, segments=segments)
    assert mesh.triangles.shape == (3 * (8 * segments + 4), 3, 3)
    assert mesh.triangles.dtype == np.float64
    surfaces = _group_surfaces(mesh)
    assert [s.id for s in surfaces] == [(b, s) for b in range(3) for s in range(4)]
    assert sum(len(s.endpoints) for s in surfaces) == 24
    hub = turbine.position + [0, 0, turbine.tower_height]
    axes = []
    for blade in range(3):
        rotation = frame(turbine, blade)
        axes.append(rotation[:, 2])
        triangles = mesh.triangles[mesh.blade_ids == blade]
        local = (triangles - hub) @ rotation
        assert_allclose(local.min(axis=(0, 1)), [-0.5, -1.5, 0], atol=1e-12)
        assert_allclose(local.max(axis=(0, 1)), [0.5, 1.5, 50], atol=1e-12)
        centers = triangles.mean(axis=1)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        assert np.all(np.einsum("ij,ij->i", normals, centers - hub - 25 * rotation[:, 2]) > 0)
        assert_allclose(np.linalg.norm(normals, axis=1), 1)
        caps = np.abs(normals @ rotation[:, 2]) > 1 - 1e-8
        assert caps.sum() == 4
        offsets = [[0, -1.5, 0], [0.5, 0, 0], [0, 1.5, 0], [-0.5, 0, 0]]
        local_normals = [[0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]]
        for side, offset in enumerate(offsets):
            surface = surfaces[blade * 4 + side]
            expected = np.array([offset, np.array(offset) + [0, 0, 50]]) @ rotation.T + hub
            assert_allclose(surface.endpoints, expected, atol=2e-12)
            assert_allclose(surface.normal, rotation @ local_normals[side], atol=2e-14)
            selected = np.linalg.norm(normals - surface.normal, axis=1) < 1e-8
            assert selected.sum() == 2 * segments
            assert_allclose((centers[selected] - surface.endpoints[0]) @ surface.normal, 0, atol=1e-12)
    assert_allclose(np.array(axes) @ np.array(axes).T, np.eye(3) * 1.5 - 0.5, atol=1e-14)


def test_rigid_transform_and_rotor_about_hub(turbine):
    original = _group_surfaces(generate_mesh(turbine, segments=3))
    moved = replace(turbine, yaw=0.6, position=[8, 4, -3])
    rotated = replace(turbine, rotor_angle=2 * np.pi / 3)
    for transformed in (moved, rotated):
        rotation = frame(transformed, 0)
        hub = np.array([0, 0, turbine.tower_height])
        result = _group_surfaces(generate_mesh(transformed, segments=3))
        for before, after in zip(original, result):
            assert before.id == after.id
            assert_allclose(after.endpoints, (before.endpoints - hub) @ rotation.T + hub + transformed.position, atol=1e-12)
            assert_allclose(after.normal, rotation @ before.normal, atol=1e-14)


@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi / 2 + 1e-12])
def test_density_order_and_repeatability(turbine, angle):
    turbine = replace(turbine, rotor_angle=angle)
    reference = _group_surfaces(generate_mesh(turbine, segments=1))
    for segments in (2, 5, 19):
        mesh = generate_mesh(turbine, segments=segments)
        permutation = np.random.default_rng(42).permutation(len(mesh.triangles))
        shuffled = Mesh(mesh.triangles[permutation], mesh.blade_ids[permutation], turbine)
        first, second = _group_surfaces(mesh), _group_surfaces(shuffled)
        for a, b, c in zip(reference, first, second):
            assert a.id == b.id == c.id
            assert_allclose(a.endpoints, b.endpoints, atol=1e-12)
            assert_allclose(b.endpoints, c.endpoints, atol=0, rtol=0)
            assert_allclose(b.normal, c.normal, atol=0, rtol=0)


def test_two_downward_sides_and_rotation_ids(turbine):
    surfaces = _group_surfaces(generate_mesh(turbine, segments=2))
    downward = [s.id for s in surfaces if s.normal[2] < -1e-8]
    assert downward == [(1, 0), (2, 2)]
    rotated = _group_surfaces(generate_mesh(replace(turbine, rotor_angle=2 * np.pi / 3), segments=2))
    assert [s.id for s in rotated] == [s.id for s in surfaces]


@pytest.mark.parametrize("scale", [1e-5, 1, 1e5])
def test_scaling(turbine, scale):
    baseline = _group_surfaces(generate_mesh(turbine, segments=4))
    scaled = Turbine(50 * scale, 3 * scale, 120 * scale)
    for a, b in zip(baseline, _group_surfaces(generate_mesh(scaled, segments=4))):
        assert_allclose(b.endpoints / scale, a.endpoints, atol=1e-12)
        assert_allclose(b.normal, a.normal, atol=1e-12)


@pytest.mark.parametrize("name", ["blade_length", "blade_width", "tower_height"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, True, [1], None])
def test_invalid_dimensions(turbine, name, value):
    with pytest.raises(ValueError, match=name):
        replace(turbine, **{name: value})


@pytest.mark.parametrize("name", ["yaw", "rotor_angle"])
@pytest.mark.parametrize("value", [np.nan, np.inf, [0], None])
def test_invalid_angles(turbine, name, value):
    with pytest.raises(ValueError, match=name):
        replace(turbine, **{name: value})


@pytest.mark.parametrize("value", [[0, 0], [[0, 0, 0]], [0, np.nan, 0], [0, 0, np.inf], "bad"])
def test_invalid_positions(turbine, value):
    with pytest.raises(ValueError, match="position"):
        replace(turbine, position=value)


@pytest.mark.parametrize("segments", [0, -1, 1.0, True, np.bool_(True), np.nan, "2", None])
def test_invalid_segments(turbine, segments):
    with pytest.raises(ValueError, match="segments"):
        generate_mesh(turbine, segments=segments)


def test_immutable_copies(turbine):
    position = np.zeros(3)
    model = replace(turbine, position=position)
    position[0] = 3
    assert model.position[0] == 0
    with pytest.raises(FrozenInstanceError):
        model.yaw = 2
    with pytest.raises(ValueError):
        model.position[0] = 2
    original = generate_mesh(model, segments=np.int64(2))
    triangles, ids = original.triangles.copy(), original.blade_ids.copy()
    mesh = Mesh(triangles, ids, model)
    triangles[:] = 0
    ids[:] = 2
    assert_allclose(mesh.triangles, original.triangles)
    assert_allclose(mesh.blade_ids, original.blade_ids)
    before = mesh.triangles.copy()
    _group_surfaces(mesh)
    assert_allclose(mesh.triangles, before, atol=0, rtol=0)


@pytest.mark.parametrize("bad", [np.zeros((0, 3, 3)), np.zeros((3, 3)), np.zeros((1, 3, 3)), np.full((1, 3, 3), np.nan)])
def test_invalid_triangles(turbine, bad):
    with pytest.raises(ValueError, match="triangles"):
        Mesh(bad, np.zeros(len(bad)), turbine)


@pytest.mark.parametrize("value", [-1, 3, 0.5, np.inf])
def test_invalid_blade_ids(turbine, value):
    mesh = generate_mesh(turbine, segments=2)
    ids = np.full(len(mesh.triangles), value)
    with pytest.raises(ValueError, match="blade_ids"):
        Mesh(mesh.triangles, ids, turbine)
    with pytest.raises(ValueError, match="blade_ids"):
        Mesh(mesh.triangles, [0], turbine)


@pytest.mark.parametrize("damage", ["missing_blade", "missing_side", "missing_triangle", "reversed", "off_plane", "extra_plane", "missing_strip", "duplicate_triangle"])
def test_abnormal_groups_rejected(turbine, damage):
    mesh = generate_mesh(turbine, segments=3)
    triangles, ids = mesh.triangles.copy(), mesh.blade_ids.copy()
    keep = np.ones(len(ids), dtype=bool)
    if damage == "missing_blade":
        keep = ids != 2
    elif damage == "missing_side":
        keep[:6] = False
    elif damage == "missing_triangle":
        keep[0] = False
    elif damage == "missing_strip":
        keep[:2] = False
    elif damage == "reversed":
        triangles[0] = triangles[0, ::-1]
    elif damage == "off_plane":
        triangles[:6, :, 1] += 0.1
    elif damage == "extra_plane":
        triangles[0, :, 1] += 0.1
    elif damage == "duplicate_triangle":
        triangles[0] = triangles[1]
    with pytest.raises(ValueError):
        _group_surfaces(Mesh(triangles[keep], ids[keep], turbine))


def test_invalid_object_types():
    with pytest.raises(ValueError, match="turbine"):
        generate_mesh(None, segments=1)
    with pytest.raises(ValueError, match="mesh"):
        _group_surfaces(None)


@pytest.mark.parametrize("position", [[500000, 6000000, 0], [6000000] * 3])
def test_dense_translated_geometry(position):
    model = Turbine(50, 3, 120, yaw=.37, rotor_angle=.29)
    original = _group_surfaces(generate_mesh(model, segments=1000))
    moved = _group_surfaces(generate_mesh(replace(model, position=position), segments=1000))
    # Coordinate roundoff in metres, and its amplification over a 5 cm strip.
    length_tol = 32 * np.finfo(float).eps * max(position)
    normal_tol = length_tol / (model.blade_length / 1000)
    assert [s.id for s in moved] == [(b, s) for b in range(3) for s in range(4)]
    for before, after in zip(original, moved):
        assert_allclose(after.endpoints - position, before.endpoints, rtol=0, atol=length_tol)
        assert_allclose(after.normal, before.normal, rtol=0, atol=normal_tol)
        assert_allclose(np.linalg.norm(after.normal), 1, rtol=0, atol=1e-14)
        assert abs((after.endpoints[1] - after.endpoints[0]) @ after.normal) <= 50 * normal_tol


@pytest.mark.parametrize("position", [[500000, 6000000, 0], [6000000] * 3])
@pytest.mark.parametrize("damage", ["reversed", "off_plane", "tilted"])
def test_dense_translated_malformed_geometry(position, damage):
    model = Turbine(50, 3, 120, position=position, yaw=.37, rotor_angle=.29)
    mesh = generate_mesh(model, segments=1000)
    triangles = mesh.triangles.copy()
    outward = -frame(model, 0)[:, 1]
    if damage == "reversed":
        triangles[0] = triangles[0, ::-1]
    elif damage == "off_plane":
        triangles[:2000] += 0.001 * outward
    else:
        triangles[0, 0] += 0.001 * outward
    with pytest.raises(ValueError):
        _group_surfaces(Mesh(triangles, mesh.blade_ids, model))
