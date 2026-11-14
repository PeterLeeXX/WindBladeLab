"""Independent full-route enumeration checks for the paired-endpoint solver."""

from dataclasses import replace
from itertools import permutations, product
from math import ceil, dist, fsum

import numpy as np
import pytest

from planning.vtnmpc import Plan, Turbine, generate_mesh, plan, plan_inspection
from planning.vtnmpc.geometry import Surface, _DIRECTION_TOL, _frame, _group_surfaces
from planning.vtnmpc.planner import _solve


def surface(i, root, tip):
    return Surface((i // 4, i % 4), np.array([root, tip], dtype=float),
                   np.array([0., 0., 1.]))


def route_distance(route):
    points = [s.endpoints if d == 0 else s.endpoints[::-1] for s, d in route]
    points = np.concatenate(points)
    return sum(dist(a, b) for a, b in zip(points, points[1:]))


def exhaustive(surfaces):
    # Deliberately no subproblem cache or dynamic programming.
    return min(route_distance(tuple(zip(order, directions)))
               for order in permutations(surfaces)
               for directions in product((0, 1), repeat=len(surfaces)))


@pytest.mark.parametrize("m", range(1, 6))
def test_matches_independent_exhaustive(m):
    rng = np.random.default_rng(182 + m)
    surfaces = [surface(i, *rng.normal(size=(2, 3))) for i in range(m)]
    route, distance = _solve(surfaces)
    assert isinstance(route, tuple)
    assert {id(s) for s, _ in route} == {id(s) for s in surfaces}
    assert len(route) == m
    assert all(d in (0, 1) for _, d in route)
    assert distance == pytest.approx(exhaustive(surfaces), rel=1e-14)
    assert distance == pytest.approx(route_distance(route), rel=1e-14)
    assert _solve(surfaces[::-1]) == (route, distance)


def test_single_surface_and_exact_stable_ties():
    surfaces = [surface(i, [0, 0, 0], [1, 0, 0]) for i in range(3)]
    assert _solve(surfaces[:1]) == (((surfaces[0], 0),), 1.)
    expected = ((surfaces[0], 0), (surfaces[1], 1), (surfaces[2], 0))
    for order in permutations(surfaces):
        assert _solve(order) == (expected, 3.)


def test_reversal_and_strictly_smaller_near_tie():
    # A 1e-12 improvement must not be swallowed by a tie tolerance.
    a = surface(0, [0, 0, 0], [1, 0, 0])
    b = surface(1, [2, 0, 0], [2 - 1e-12, 0, 0])
    route, distance = _solve([b, a])
    assert route == ((a, 0), (b, 1))
    assert distance == 2.
    assert distance < route_distance(((a, 0), (b, 0)))


def test_nearest_neighbor_trap():
    # Even greedy continuations from every possible start are insufficient.
    rng = np.random.default_rng(42)
    surfaces = [surface(i, *rng.integers(-10, 11, size=(2, 3))) for i in range(5)]
    greedy_costs = []
    for first in surfaces:
        for direction in (0, 1):
            route = [(first, direction)]
            remaining = [s for s in surfaces if s is not first]
            while remaining:
                current, d = route[-1]
                following, orientation = min(
                    product(remaining, (0, 1)),
                    key=lambda pair: dist(current.endpoints[1 - d], pair[0].endpoints[pair[1]]),
                )
                route.append((following, orientation))
                remaining.remove(following)
            greedy_costs.append(route_distance(route))
    _, optimum = _solve(surfaces)
    assert optimum == pytest.approx(exhaustive(surfaces))
    assert optimum < min(greedy_costs)


def test_cache_isolation_and_no_mutation():
    surfaces = [surface(1, [3, 0, 0], [4, 0, 0]),
                surface(0, [0, 0, 0], [1, 0, 0])]
    original_order = surfaces[:]
    originals = [(s.endpoints.copy(), s.normal.copy()) for s in surfaces]
    result = _solve(surfaces)
    scaled = [Surface(s.id, s.endpoints * 7, s.normal.copy()) for s in surfaces]
    assert _solve(scaled)[1] == result[1] * 7
    assert _solve(surfaces) == result
    assert surfaces == original_order
    for s, (endpoints, normal) in zip(surfaces, originals):
        np.testing.assert_array_equal(s.endpoints, endpoints)
        np.testing.assert_array_equal(s.normal, normal)


@pytest.mark.parametrize("points", [
    [[0, 0, 0], [np.nan, 0, 0]],
    [[0, 0, 0], [np.inf, 0, 0]],
    [[-1e308, 0, 0], [1e308, 0, 0]],
    [[0, 0], [1, 1]],
])
def test_invalid_endpoints(points):
    with pytest.raises(ValueError, match="endpoint"):
        _solve([Surface((0, 0), np.array(points), np.zeros(3))])


def test_surface_count_and_duplicate_ids():
    a = surface(0, [0, 0, 0], [1, 0, 0])
    for surfaces in ([], [a, a], [a] * 13):
        with pytest.raises(ValueError, match="surfaces"):
            _solve(surfaces)


def test_full_plan_output_contract():
    turbine = Turbine(50, 3, 120, position=[4, -3, 2], yaw=.37, rotor_angle=.29)
    mesh = generate_mesh(turbine, segments=3)
    triangles, ids = mesh.triangles.copy(), mesh.blade_ids.copy()
    result = plan(mesh, spacing=7)
    assert_plan_contract(result, mesh, spacing=7, count=12)
    np.testing.assert_array_equal(mesh.triangles, triangles)
    np.testing.assert_array_equal(mesh.blade_ids, ids)


def assert_plan_contract(result, mesh, *, spacing, count):
    turbine = mesh.turbine
    assert isinstance(result, Plan)
    assert set(vars(result)) == {"points", "normals", "surface_order", "offsets", "distance"}
    assert result.points.dtype == result.normals.dtype == np.float64
    assert result.points.shape == result.normals.shape
    assert result.points.shape[1] == 3
    assert np.isfinite(result.points).all() and np.isfinite(result.normals).all()
    np.testing.assert_allclose(np.linalg.norm(result.normals, axis=1), 1)
    assert len(result.surface_order) == len(set(result.surface_order)) == count
    assert result.offsets.shape == (count + 1,)
    assert np.issubdtype(result.offsets.dtype, np.integer)
    assert result.offsets[0] == 0 and result.offsets[-1] == len(result.points)
    surfaces = {s.id: s for s in _group_surfaces(mesh)}
    endpoints = []
    for sid, lo, hi in zip(result.surface_order, result.offsets, result.offsets[1:]):
        s = surfaces[sid]
        points, normals = result.points[lo:hi], result.normals[lo:hi]
        assert len(points) == ceil(dist(*s.endpoints) / spacing) + 1
        assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) <= spacing + 1e-12
        expected = s.endpoints if np.allclose(points[0], s.endpoints[0]) else s.endpoints[::-1]
        np.testing.assert_array_equal(points[[0, -1]], expected)
        np.testing.assert_array_equal(normals, np.tile(s.normal, (len(points), 1)))
        hub, axial, tangent, axis = _frame(turbine, sid[0])
        local = points - hub
        width = turbine.blade_width / (2 if sid[1] % 2 == 0 else 6)
        np.testing.assert_allclose(local @ s.normal, width, atol=1e-12)
        for vector, lower, upper in ((axis, 0, turbine.blade_length),
                                     (axial, -turbine.blade_width / 6, turbine.blade_width / 6),
                                     (tangent, -turbine.blade_width / 2, turbine.blade_width / 2)):
            coordinates = local @ vector
            assert coordinates.min() >= lower - 1e-12
            assert coordinates.max() <= upper + 1e-12
        endpoints.extend(points[[0, -1]])
    assert result.distance == fsum(dist(a, b) for a, b in zip(endpoints, endpoints[1:]))


def test_subset_order_repeatability_and_spacing_independent_of_mesh():
    turbine = Turbine(11, 3, 20, yaw=.23, rotor_angle=.18)
    selected = [(2, 3), (0, 1), (1, 2)]
    original = selected[:]
    mesh = generate_mesh(turbine, segments=2)
    first = plan(mesh, spacing=2.4, surface_ids=selected)
    for subset in (selected, selected[::-1], tuple(selected)):
        other = plan(mesh, spacing=2.4, surface_ids=subset)
        assert other.surface_order == first.surface_order
        assert other.distance == first.distance
        np.testing.assert_array_equal(other.points, first.points)
        np.testing.assert_array_equal(other.normals, first.normals)
        np.testing.assert_array_equal(other.offsets, first.offsets)
    assert selected == original
    # A single face fixes orientation, so mesh-density roundoff cannot change ties.
    for segments in (1, 3, 8):
        other = plan(generate_mesh(turbine, segments=segments), spacing=2.4,
                     surface_ids=[(0, 1)])
        reference = plan(mesh, spacing=2.4, surface_ids=[(0, 1)])
        np.testing.assert_allclose(other.points, reference.points, atol=1e-12)
        np.testing.assert_allclose(other.normals, reference.normals, atol=1e-12)
        np.testing.assert_array_equal(other.offsets, reference.offsets)
        assert other.distance == pytest.approx(reference.distance)
    coarse = plan(mesh, spacing=100, surface_ids=selected)
    assert np.all(np.diff(coarse.offsets) == 2)
    assert coarse.surface_order == first.surface_order
    assert coarse.distance == first.distance


def test_objective_rigid_motion_and_scaling():
    selected = [(0, 0), (0, 3), (1, 2), (2, 1)]
    baseline = plan(generate_mesh(Turbine(11, 3, 20), segments=2),
                    spacing=2, surface_ids=selected)
    for scale, position, yaw, rotor in ((1, [7, -4, 3], .63, .49),
                                       (3.7, [0, 0, 0], 0, 0)):
        mesh = generate_mesh(Turbine(11 * scale, 3 * scale, 20 * scale,
                                    position=position, yaw=yaw, rotor_angle=rotor), segments=2)
        result = plan(mesh, spacing=2 * scale, surface_ids=selected)
        assert result.distance == pytest.approx(baseline.distance * scale, rel=1e-13)


@pytest.mark.parametrize("spacing", [0, -1, np.nan, np.inf, -np.inf, True, None, [1], np.array([1, 2]), "bad"])
def test_invalid_spacing(spacing):
    mesh = generate_mesh(Turbine(10, 3, 20), segments=1)
    with pytest.raises(ValueError, match="spacing"):
        plan(mesh, spacing=spacing)


@pytest.mark.parametrize("subset", [[], (), 1, [(0, 0), (0, 0)], [(3, 0)], [(0, 4)],
                                     [(-1, 0)], [(0., 1)], [(True, 1)], [(0,)],
                                     [[0, 1]], np.array([[0, 1]]), [np.array([0, 1])],
                                     [(np.nan, 0)], [(0, 1, 2)], "bad"])
def test_invalid_surface_ids(subset):
    mesh = generate_mesh(Turbine(10, 3, 20), segments=1)
    with pytest.raises(ValueError, match="surface_ids"):
        plan(mesh, spacing=2, surface_ids=subset)


def test_invalid_mesh_and_numpy_integer_ids():
    with pytest.raises(ValueError, match="mesh"):
        plan(np.zeros((2, 3)), spacing=2)
    mesh = generate_mesh(Turbine(10, 3, 20), segments=1)
    result = plan(mesh, spacing=2, surface_ids=[(np.int64(0), np.int32(1))])
    assert result.surface_order == ((0, 1),)


def test_shared_endpoint_retains_separate_face_normals(monkeypatch):
    from importlib import import_module

    planner = import_module("planning.vtnmpc.planner")
    a = surface(0, [0, 0, 0], [1, 0, 0])
    b = Surface((0, 1), np.array([[1., 0, 0], [2., 0, 0]]), np.array([0., 1, 0]))
    monkeypatch.setattr(planner, "_group_surfaces", lambda mesh: (a, b))
    result = plan(None, spacing=.6)
    assert result.surface_order == ((0, 0), (0, 1))
    np.testing.assert_array_equal(result.offsets, [0, 3, 6])
    np.testing.assert_array_equal(result.points[2], result.points[3])
    np.testing.assert_array_equal(result.normals[2], a.normal)
    np.testing.assert_array_equal(result.normals[3], b.normal)
    assert result.distance == 2


@pytest.mark.parametrize("angle", [0, 2 * np.pi, -4 * np.pi,
                                  _DIRECTION_TOL / 4, -_DIRECTION_TOL / 4])
def test_two_stage_paper_dimensions_and_physical_identity(angle):
    turbine = Turbine(50, 3, 120, position=[4, -3, 2], yaw=.37, rotor_angle=angle)
    initial = generate_mesh(turbine, segments=3)
    rotated = generate_mesh(replace(turbine, rotor_angle=(angle % (2 * np.pi))
                                    + 2 * np.pi / 3), segments=3)
    stages = plan_inspection(turbine, segments=3, spacing=7)
    assert isinstance(stages, tuple) and len(stages) == 2
    first, second = stages
    assert first is not second
    assert_plan_contract(first, initial, spacing=7, count=10)
    assert_plan_contract(second, rotated, spacing=7, count=2)
    original = {s.id: s for s in _group_surfaces(initial)}
    deferred = {sid for sid, s in original.items() if s.normal[2] < -_DIRECTION_TOL}
    assert deferred == {(1, 0), (2, 2)}
    assert set(first.surface_order).isdisjoint(second.surface_order)
    assert set(first.surface_order) | set(second.surface_order) == set(original)
    assert set(second.surface_order) == deferred
    for sid, lo, hi in zip(second.surface_order, second.offsets, second.offsets[1:]):
        old = original[sid]
        assert not np.allclose(second.normals[lo], old.normal)
        assert not np.allclose(second.points[[lo, hi - 1]], old.endpoints)
        assert not np.allclose(second.points[[lo, hi - 1]], old.endpoints[::-1])
    assert second.distance == pytest.approx(exhaustive(
        [s for s in _group_surfaces(rotated) if s.id in deferred]))
    assert turbine.rotor_angle == angle


@pytest.mark.parametrize("angle", [np.pi, .1, -0.1, 2 * np.pi / 3,
                                  -2 * np.pi / 3, 4 * np.pi / 3,
                                  2 * _DIRECTION_TOL, -2 * _DIRECTION_TOL])
def test_inspection_rejects_nonzero_initial_blade_angle(angle):
    with pytest.raises(ValueError, match="rotor_angle"):
        plan_inspection(Turbine(50, 3, 120, rotor_angle=angle), segments=2, spacing=7)


@pytest.mark.parametrize("kwargs, parameter", [
    ({"turbine": None}, "turbine"),
    ({"segments": 0}, "segments"),
    ({"segments": True}, "segments"),
    ({"spacing": 0}, "spacing"),
    ({"spacing": np.nan}, "spacing"),
])
def test_inspection_invalid_inputs(kwargs, parameter):
    arguments = dict(turbine=Turbine(50, 3, 120), segments=2, spacing=7)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=parameter):
        plan_inspection(**arguments)


@pytest.mark.parametrize("position", [[500000, 6000000, 0], [6000000] * 3])
def test_dense_translated_public_planning(position):
    model = Turbine(50, 3, 120, yaw=.37, rotor_angle=.29)
    original = generate_mesh(model, segments=1000)
    moved = generate_mesh(replace(model, position=position), segments=1000)
    length_tol = 32 * np.finfo(float).eps * max(position)
    for subset in ([(0, 0)], None):
        reference = plan(original, spacing=7, surface_ids=subset)
        result = plan(moved, spacing=7, surface_ids=subset)
        assert set(result.surface_order) == set(reference.surface_order)
        # At most 23 graph edges, each with two rounded endpoints.
        assert result.distance == pytest.approx(reference.distance, rel=0, abs=46 * length_tol)
        assert np.isfinite(result.points).all()
        if subset is not None:
            np.testing.assert_allclose(result.points - position, reference.points,
                                       rtol=0, atol=length_tol)
            np.testing.assert_allclose(result.normals, reference.normals,
                                       rtol=0, atol=length_tol / .05)
    upright = replace(model, rotor_angle=0)
    reference = plan_inspection(upright, segments=1000, spacing=7)
    stages = plan_inspection(replace(upright, position=position), segments=1000, spacing=7)
    for before, after in zip(reference, stages):
        assert set(after.surface_order) == set(before.surface_order)
        assert after.distance == pytest.approx(before.distance, rel=0, abs=46 * length_tol)
        assert np.isfinite(after.points).all() and np.isfinite(after.normals).all()
    assert [len(stage.surface_order) for stage in stages] == [10, 2]
