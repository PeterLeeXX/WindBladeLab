"""Exact open-route ordering of paired surface endpoints."""

from dataclasses import dataclass, replace
from functools import cache
from math import ceil, dist, fsum, isclose, isfinite
from numbers import Integral

import numpy as np

from .geometry import (Surface, Turbine, _DIRECTION_TOL, _group_surfaces,
                       _scalar, generate_mesh)


def _solve(surfaces) -> tuple[tuple[tuple[Surface, int], ...], float]:
    """Return (tuple of (Surface, direction), distance); 0 is root to tip.

    Enumerate all starts and continuations with a cache local to this solve.
    Exact ties retain the first stable-ID/direction-ordered candidate.
    """
    surfaces = tuple(sorted(surfaces, key=lambda surface: surface.id))
    m = len(surfaces)
    if not 1 <= m <= 12:
        raise ValueError("surfaces must contain between 1 and 12 surfaces")
    if len({surface.id for surface in surfaces}) != m:
        raise ValueError("surfaces must have unique IDs")
    endpoints = []
    for surface in surfaces:
        points = np.asarray(surface.endpoints, dtype=float)
        if points.shape != (2, 3) or not np.all(np.isfinite(points)):
            raise ValueError("surface endpoints must have finite shape (2, 3)")
        endpoints.extend(points)
    distances = tuple(tuple(dist(a, b) for b in endpoints) for a in endpoints)
    if not all(isfinite(value) for row in distances for value in row):
        raise ValueError("surface endpoint distances must be finite")
    lengths = tuple(distances[2 * i][2 * i + 1] for i in range(m))
    full = (1 << m) - 1

    @cache
    def visit(mask, current, direction):
        if mask == full:
            return 0.0, None
        best, successor = float("inf"), None
        exit_node = 2 * current + 1 - direction
        for following in range(m):
            if mask & (1 << following):
                continue
            for orientation in (0, 1):
                remainder, _ = visit(mask | (1 << following), following, orientation)
                cost = (distances[exit_node][2 * following + orientation]
                        + lengths[following] + remainder)
                if cost < best:
                    best, successor = cost, (following, orientation)
        return best, successor

    best, start = float("inf"), None
    for current in range(m):
        for direction in (0, 1):
            remainder, _ = visit(1 << current, current, direction)
            cost = lengths[current] + remainder
            if cost < best:
                best, start = cost, (current, direction)
    if not isfinite(best):
        raise ValueError("surface route distance must be finite")
    route, mask = [], 0
    step = start
    while step is not None:
        current, direction = step
        route.append((surfaces[current], direction))
        mask |= 1 << current
        _, step = visit(mask, current, direction)
    return tuple(route), best


@dataclass
class Plan:
    """Segmented surface references, not a safe flight trajectory.

    Distance is the endpoint-graph cost of a free-start, free-end open route,
    including inter-face connections but no takeoff, return, or safe transfers.
    Offsets delimit each face's points and outward unit normals.
    """

    points: np.ndarray
    normals: np.ndarray
    surface_order: tuple[tuple[int, int], ...]
    offsets: np.ndarray
    distance: float


def plan(mesh, *, spacing, surface_ids=None) -> Plan:
    """Exactly order generated mesh surfaces and sample each face separately.

    Spacing bounds only within-face sample gaps, independently of mesh density.
    Surface IDs are unique integer (blade, side) tuples; None selects all faces.
    The free open route contains surface references, not safe flight transfers.
    """
    spacing = _scalar(spacing, "spacing", positive=True)
    surfaces = _group_surfaces(mesh)
    if surface_ids is not None:
        try:
            ids = tuple(surface_ids)
        except TypeError:
            raise ValueError("surface_ids must be a nonempty iterable of ID tuples") from None
        valid = {surface.id for surface in surfaces}
        if not ids or any(
            not isinstance(sid, tuple) or len(sid) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in sid)
            for sid in ids
        ):
            raise ValueError("surface_ids must contain integer (blade, side) tuples")
        selected = set(ids)
        if len(selected) != len(ids) or not selected <= valid:
            raise ValueError("surface_ids must be valid and unique")
        surfaces = tuple(surface for surface in surfaces if surface.id in selected)
    route, optimum = _solve(surfaces)
    points, normals, offsets, endpoints = [], [], [0], []
    for surface, direction in route:
        start, end = surface.endpoints[::1 if direction == 0 else -1]
        ratio = dist(start, end) / spacing
        if not isfinite(ratio):
            raise ValueError("spacing is too small for finite sampling intervals")
        intervals = max(1, ceil(ratio))
        samples = np.linspace(start, end, intervals + 1)
        points.append(samples)
        normals.append(np.tile(surface.normal, (len(samples), 1)))
        offsets.append(offsets[-1] + len(samples))
        endpoints.extend((start, end))
    distance = fsum(dist(a, b) for a, b in zip(endpoints, endpoints[1:]))
    # Validation only: the solver's strict comparisons alone select the route.
    if not isclose(distance, optimum, rel_tol=1e-13, abs_tol=0.0):
        raise ValueError("surface route distance disagrees with solver")
    return Plan(np.concatenate(points), np.concatenate(normals),
                tuple(surface.id for surface, _ in route),
                np.array(offsets, dtype=np.int64), distance)


def plan_inspection(turbine, *, segments, spacing) -> tuple[Plan, Plan]:
    """Plan inverted-Y inspection as ten faces, then two after +120 degrees.

    Blade zero must initially point up (whole turns are equivalent). Physical
    surface IDs persist through rotation. No rotation-time transfer is planned;
    the two independent optima are not a whole-mission flight optimum.
    """
    if not isinstance(turbine, Turbine):
        raise ValueError("turbine must be a Turbine")
    angle = turbine.rotor_angle % (2 * np.pi)
    if np.hypot(np.sin(angle), np.cos(angle) - 1) > _DIRECTION_TOL:
        raise ValueError("rotor_angle must be zero modulo whole turns for inverted-Y inspection")
    spacing = _scalar(spacing, "spacing", positive=True)
    mesh = generate_mesh(turbine, segments=segments)
    surfaces = _group_surfaces(mesh)
    deferred = tuple(s.id for s in surfaces if s.normal[2] < -_DIRECTION_TOL)
    if len(deferred) != 2:
        raise ValueError("rotor_angle must yield exactly two downward-facing surfaces")
    first = plan(mesh, spacing=spacing,
                 surface_ids=tuple(s.id for s in surfaces if s.id not in deferred))
    rotated = replace(turbine, rotor_angle=angle + 2 * np.pi / 3)
    second = plan(generate_mesh(rotated, segments=segments), spacing=spacing,
                  surface_ids=deferred)
    return first, second
