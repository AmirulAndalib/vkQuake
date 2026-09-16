#!/usr/bin/env python3
"""Exercise vkQuake's gravity patch without building or running the engine.

Ports the pre-patch and paired-velocity SV_FlyMove paths, including float32
storage/arithmetic, four bumps, plane resets, STOP_EPSILON, and impact order.
Geometry is a point swept through unions of convex solid brushes, interpreted
as already expanded collision hulls. This is NOT a BSP tracer, QuakeC VM,
pusher simulation, or proof of engine parity. FMA/compiler differences are
also outside this model. The legacy and paired movers are separate ports.

Run from any directory: python Misc/physics_gravity_sim.py
Requires numpy and matplotlib.
Outputs go to the ignored build/physics-gravity-sim directory by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

F = np.float32
H = 1.0 / 72.0
EPS = 1.0 / 32.0
RATES = (30, 40, 58, 60, 72, 77, 120, 144, 240)
ZERO = np.zeros(3, dtype=np.float32)
COUNTS = Counter()


def vec(x):
    return np.array(x, dtype=np.float32)


def dot(a, b):
    return F(F(F(a[0] * b[0]) + F(a[1] * b[1])) + F(a[2] * b[2]))


def ma(a, t, b):
    return a + F(t) * b


def clip(v, n, bounce=1.0):
    result = v - n * F(dot(v, n) * F(bounce))
    result[(result > -0.1) & (result < 0.1)] = 0.0
    return result


@dataclass
class Plane:
    normal: np.ndarray
    dist: float = 0.0

    def distance(self, p):
        # Quake's non-axial hull plane tests use double-precision dot products.
        return float(np.dot(self.normal.astype(float), p.astype(float))) - self.dist


@dataclass
class Brush:
    planes: list[Plane]
    name: str

    def depth(self, p):
        return max(plane.distance(p) for plane in self.planes)


@dataclass
class Trace:
    fraction: np.float32
    endpos: np.ndarray
    normal: np.ndarray = field(default_factory=lambda: ZERO.copy())
    name: str = ""
    allsolid: bool = False
    startsolid: bool = False


class World:
    def __init__(self, *brushes):
        self.brushes = brushes

    def trace(self, start, end):
        best = Trace(F(1), end.copy())
        for brush in self.brushes:
            ds = [p.distance(start) for p in brush.planes]
            de = [p.distance(end) for p in brush.planes]
            if all(d < 0 for d in ds):
                best.startsolid = True
                if all(d < 0 for d in de):
                    return Trace(F(0), start.copy(), name=brush.name,
                                 allsolid=True, startsolid=True)
                continue
            enter, leave, hit = -1.0, 1.0, None
            for i, (s, e) in enumerate(zip(ds, de)):
                if s >= 0 and e >= 0:
                    hit = None
                    break
                if s < 0 and e < 0:
                    continue
                crossing = s / (s - e)
                if s >= 0:
                    if crossing > enter:
                        enter, hit = crossing, i
                else:
                    leave = min(leave, crossing)
            else:
                if hit is not None and enter <= leave:
                    fraction = F(max(0.0, (ds[hit] - EPS) / (ds[hit] - de[hit])))
                    if fraction < best.fraction:
                        best = Trace(fraction, ma(start, fraction, end - start),
                                     brush.planes[hit].normal, brush.name,
                                     startsolid=best.startsolid)
        return best

    def penetration(self, p):
        return max((max(0.0, -b.depth(p)) for b in self.brushes), default=0.0)


def halfspace(normal, dist=0, name="plane"):
    return Brush([Plane(vec(normal), dist)], name)


def box(lo, hi, name="box"):
    planes = []
    for axis in range(3):
        n = ZERO.copy()
        n[axis] = 1
        planes.extend((Plane(n, hi[axis]), Plane(-n, -lo[axis])))
    return Brush(planes, name)


@dataclass
class Entity:
    pos: np.ndarray
    vel: np.ndarray
    ground: bool = False
    thud: bool = False
    free: bool = False
    gravity: float = 800.0
    touches: list = field(default_factory=list)
    last_wall: Trace | None = None

    def clone(self):
        return Entity(self.pos.copy(), self.vel.copy(), self.ground, self.thud,
                      self.free, self.gravity, list(self.touches))


def impact(ent, trace, callback):
    ent.touches.append((trace.name, trace.normal.copy()))
    if callback:
        callback(ent, trace)


def thud_touch(ent, trace):
    if ent.ground:
        ent.ground = False
        ent.thud = True


def fly_old(ent, time, world, callback=None):
    """Pre-patch SV_FlyMove, without pusher support-plane rewriting."""
    original = ent.vel.copy()
    primal = ent.vel.copy()
    planes = []
    left = F(time)
    blocked = 0
    for _ in range(4):
        if not np.any(ent.vel):
            break
        trace = world.trace(ent.pos, ma(ent.pos, left, ent.vel))
        if trace.allsolid:
            ent.vel = ZERO.copy()
            return 3
        if trace.fraction > 0:
            ent.pos = trace.endpos.copy()
            original = ent.vel.copy()
            planes = []
        if trace.fraction == 1:
            break
        if trace.normal[2] > F(0.7):
            blocked |= 1
            ent.ground = True
        if trace.normal[2] == 0:
            blocked |= 2
            ent.last_wall = trace
        impact(ent, trace, callback)
        if ent.free:
            break
        left = F(left - F(left * trace.fraction))
        if len(planes) >= 5:
            ent.vel = ZERO.copy()
            return 3
        planes.append(trace.normal)
        for i, plane in enumerate(planes):
            new = clip(original, plane)
            if all(j == i or dot(new, other) >= 0 for j, other in enumerate(planes)):
                ent.vel = new
                break
        else:
            COUNTS["old_crease"] += 1
            if len(planes) != 2:
                ent.vel = ZERO.copy()
                return 7
            direction = np.cross(planes[0], planes[1])
            ent.vel = direction * dot(direction, ent.vel)
        if dot(ent.vel, primal) <= 0:
            ent.vel = ZERO.copy()
            return blocked
    return blocked


def fly_paired(ent, time, world, move_velocity, callback=None):
    """Current working-tree SV_FlyMove, with separate sweep/end velocities."""
    sweep = move_velocity.copy()
    original = sweep.copy()
    primal = sweep.copy()
    original_end = ent.vel.copy()
    planes = []
    left = F(time)
    blocked = 0
    for _ in range(4):
        if not np.any(sweep):
            break
        trace = world.trace(ent.pos, ma(ent.pos, left, sweep))
        if trace.allsolid:
            ent.vel = ZERO.copy()
            return 3
        if trace.fraction > 0:
            ent.pos = trace.endpos.copy()
            original = sweep.copy()
            original_end = ent.vel.copy()
            planes = []
        if trace.fraction == 1:
            break
        if trace.normal[2] > F(0.7):
            blocked |= 1
            ent.ground = True
        if trace.normal[2] == 0:
            blocked |= 2
            ent.last_wall = trace
        impact_velocity = ent.vel.copy()
        impact(ent, trace, callback)
        if ent.free:
            break
        if not np.array_equal(ent.vel, impact_velocity):
            sweep = ent.vel.copy()
        left = F(left - F(left * trace.fraction))
        if len(planes) >= 5:
            ent.vel = ZERO.copy()
            return 3
        planes.append(trace.normal)
        for i, plane in enumerate(planes):
            new = clip(original, plane)
            new_end = clip(original_end, plane)
            if all(j == i or (dot(new, other) >= 0 and dot(new_end, other) >= 0)
                   for j, other in enumerate(planes)):
                sweep, ent.vel = new, new_end
                break
        else:
            COUNTS["paired_crease"] += 1
            if len(planes) != 2:
                ent.vel = ZERO.copy()
                return 7
            direction = np.cross(planes[0], planes[1])
            sweep = direction * dot(direction, sweep)
            ent.vel = direction * dot(direction, ent.vel)
        if dot(sweep, primal) <= 0:
            ent.vel = ZERO.copy()
            return blocked
    return blocked


def add_gravity(ent, dt, mode, analytic=True):
    move_time = (dt + H) * 0.5 if analytic else dt
    sweep = ent.vel.copy()
    sweep[2] = F(float(sweep[2]) - ent.gravity * move_time)
    stored_time = dt if mode == "paired" else move_time
    ent.vel[2] = F(float(ent.vel[2]) - ent.gravity * stored_time)
    return sweep


def finish_old(ent, dt, analytic):
    if analytic and not ent.ground and not ent.free:
        ent.vel[2] = F(float(ent.vel[2]) - ent.gravity * (dt - H) * 0.5)


def tick(ent, dt, world, mode, callback=None, link_callback=None, analytic=True):
    """One active STEP fall, including the final SV_LinkEdict callback slot.

    Callers explicitly manage ground state to select active movement, as QC or
    player walking does. Water, thinks, and landing sounds are not simulated.
    """
    sweep = add_gravity(ent, dt, mode, analytic)
    ent.vel = np.clip(ent.vel, -2000, 2000).astype(np.float32)
    sweep = np.clip(sweep, -2000, 2000).astype(np.float32)
    if mode == "paired":
        blocked = fly_paired(ent, dt, world, sweep, callback)
    else:
        blocked = fly_old(ent, dt, world, callback)
    if not ent.free and link_callback:
        link_callback(ent)
    if mode != "paired":
        finish_old(ent, dt, analytic)
    return blocked


def toss_tick(ent, dt, world, mode, callback=None, bounce=True):
    if ent.ground or ent.free:
        return
    sweep = add_gravity(ent, dt, mode)
    trace = world.trace(ent.pos, ma(ent.pos, dt, sweep))
    ent.pos = trace.endpos.copy()
    if trace.fraction < 1:
        impact(ent, trace, callback)
    if ent.free:
        return
    if mode != "paired":
        finish_old(ent, dt, True)
    if trace.fraction == 1:
        return
    ent.vel = clip(ent.vel, trace.normal, 1.5 if bounce else 1)
    if trace.normal[2] > F(0.7) and (not bounce or dot(ent.vel, trace.normal) < 60):
        ent.ground = True
        ent.vel = ZERO.copy()


def walkmove_partial(ent, distance, world):
    """SV_movestep's downward sweep with Frogbot's FL_PARTIALGROUND set.

    No four-corner support check is needed for these broad, flat step tops.
    """
    new = ent.pos + vec((distance, 0, 18))
    end = new - vec((0, 0, 36))
    trace = world.trace(new, end)
    if trace.allsolid:
        return False
    if trace.startsolid:
        new[2] -= F(18)
        trace = world.trace(new, end)
        if trace.allsolid or trace.startsolid:
            return False
    if trace.fraction == 1:
        ent.pos += vec((distance, 0, 0))
        ent.ground = False
    else:
        ent.pos = trace.endpos.copy()
    return True


def push(ent, end, world, callback):
    trace = world.trace(ent.pos, end)
    ent.pos = trace.endpos.copy()
    if trace.fraction < 1:
        impact(ent, trace, callback)
    return trace


def player_tick(ent, dt, world, mode, callback=None):
    """SV_WalkMove with no water/pushers and pr_checkextension=1.

    Includes the actual up/forward/down retry, old/no-step velocity snapshots,
    rollback, and wall friction for a fixed yaw of zero. Input acceleration is
    supplied by the test. TryUnstick is intentionally disabled, as with the
    engine's pr_checkextension=1 path.
    """
    sweep = add_gravity(ent, dt, mode)

    def fly(move):
        if mode == "paired":
            return fly_paired(ent, dt, world, move, callback)
        return fly_old(ent, dt, world, callback)

    def walk():
        was_ground = ent.ground
        ent.ground = False
        old_pos, old_vel = ent.pos.copy(), ent.vel.copy()
        blocked = fly(sweep)
        if not blocked & 2 or not was_ground:
            return
        COUNTS[mode + "_stair_retry"] += 1
        no_step_pos, no_step_vel = ent.pos.copy(), ent.vel.copy()
        ent.pos = old_pos.copy()
        push(ent, ent.pos + vec((0, 0, 18)), world, callback)
        ent.vel = vec((old_vel[0], old_vel[1], 0))
        blocked = fly(ent.vel.copy())
        if blocked & 2:
            normal = ent.last_wall.normal
            d = F(normal[0] + F(0.5))
            if d < 0:
                side = ent.vel - normal * dot(normal, ent.vel)
                ent.vel[:2] = side[:2] * F(1 + d)
        down = ent.pos.copy()
        down[2] = F(float(down[2]) - 18 + float(sweep[2]) * dt)
        trace = push(ent, down, world, callback)
        if trace.normal[2] <= F(0.7):
            COUNTS[mode + "_stair_rollback"] += 1
            ent.pos, ent.vel = no_step_pos, no_step_vel

    walk()
    if mode != "paired":
        finish_old(ent, dt, True)


def ramp(x0, x1, z0, angle, name):
    a = math.radians(angle)
    return Brush([
        Plane(vec((-math.sin(a), 0, math.cos(a))), z0 * math.cos(a) - x0 * math.sin(a)),
        Plane(vec((-1, 0, 0)), -x0), Plane(vec((1, 0, 0)), x1),
        Plane(vec((0, 0, -1)), 1000),
    ], name)


def listen_ticks(render_hz, seconds=1):
    """Ideal steady rendering, porting host.c's listen-server accumulator.

    host_maxfps <= 72 disables isolation; above it the threshold is a float
    1/71.999 and each server tick consumes min(accumtime, 0.017).
    """
    interval = float(F(1 / 71.999))
    accum = 0.0
    for _ in range(round(render_hz * seconds)):
        if render_hz <= 72:
            yield 1 / render_hz
            continue
        accum += 1 / render_hz
        while accum >= interval:
            dt = min(accum, 0.017)
            accum -= dt
            yield dt


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.failures = []
        self.failure_counts = Counter()

    def check(self, group, condition, details):
        self.counts[group] += 1
        if not condition:
            self.failure_counts[group] += 1
            if len(self.failures) < 100:
                self.failures.append({"group": group, "details": details})


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run(checks, random_cases):
    results = {}
    floor = halfspace((0, 0, 1), name="floor")
    empty = World()

    # Analytic oracle independent of either movement port.
    freefall = []
    for hz in RATES:
        ent = Entity(vec((0, 0, 100)), vec((100, -10, 270)))
        for _ in range(hz):
            ent.ground = False
            tick(ent, 1 / hz, empty, "paired")
        expected = vec((100, -10, 100 + 270 - 400 * (1 + H)))
        error = float(np.max(np.abs(ent.pos - expected)))
        checks.check("freefall", error < 0.003 and abs(ent.vel[2] + 530) < 0.003,
                     {"hz": hz, "position_error": error, "velocity": ent.vel})
        freefall.append({"hz": hz, "position_error": error})
    results["freefall"] = freefall

    frogs = []
    for hz in RATES:
        for height in (1, 8, 16, 18, 20):
            world = World(floor, box((0, -100, -100), (100, 100, height), "step"))
            for mode in ("split", "paired"):
                ent = Entity(vec((-EPS, 0, EPS)), vec((320, 0, 0)))
                start = ent.pos.copy()
                pre = ent.vel.copy()
                tick(ent, 1 / hz, world, mode, thud_touch)
                if ent.thud:
                    ent.ground, ent.thud = True, False
                delta_z = float(pre[2] - ent.vel[2])
                attempted = ent.ground and abs(delta_z) < 0.1 and ent.vel[0] < pre[0]
                moved = False
                if attempted:
                    distance = float(pre[0]) / hz - float(ent.pos[0] - start[0])
                    moved = walkmove_partial(ent, distance, world)
                    if not moved:
                        moved = walkmove_partial(ent, 0.95, world)
                row = {"hz": hz, "height": height, "mode": mode, "delta_z": delta_z,
                       "attempted": bool(attempted), "climbed": bool(moved), "z": float(ent.pos[2]),
                       "floor_contact": any(name == "floor" for name, _ in ent.touches)}
                frogs.append(row)
                if mode == "paired":
                    if row["floor_contact"]:
                        checks.check("frogbot", attempted and moved == (height <= 18), row)
                    else:
                        # At 240Hz the fall from rest is 0.03009 units, shorter
                        # than the 0.03125 collision clearance. Both old/new paths
                        # are still falling. This is not a clipped landing.
                        checks.check("floor_clearance", not attempted and not moved and hz == 240, row)
                elif hz == 60 and height == 8:
                    checks.check("reproduce_original", not attempted and abs(delta_z - 10 / 9) < 1e-5, row)
    results["frogbot"] = frogs

    schedules = []
    world = World(floor, box((0, -100, -100), (100, 100, 8), "step"))
    for render_hz in (60, 72, 77, 120, 240):
        ticks = list(listen_ticks(render_hz))
        for mode in ("split", "paired"):
            ent = Entity(vec((-EPS, 0, EPS)), vec((320, 0, 0)), ground=True)
            first_climb = None
            deltas = []
            for number, dt in enumerate(ticks, 1):
                ent.vel[0] = F(320)  # prescribed forward input, not full bot AI
                if ent.ground and ent.vel[2] < 0:
                    ent.vel[2] = F(0)
                ent.ground = False  # FrogbotPrePhysics2
                pre, origin = ent.vel.copy(), ent.pos.copy()
                tick(ent, dt, world, mode, thud_touch)
                if ent.thud:
                    ent.ground, ent.thud = True, False
                delta_z = float(pre[2] - ent.vel[2])
                deltas.append(delta_z)
                if ent.ground and abs(delta_z) < 0.1 and ent.vel[0] < pre[0]:
                    distance = float(pre[0]) * dt - float(ent.pos[0] - origin[0])
                    if walkmove_partial(ent, distance, world) or walkmove_partial(ent, 0.95, world):
                        first_climb = number
                        break
            row = {"render_fps": render_hz, "mode": mode, "physics_ticks_in_one_second": len(ticks),
                   "first_tick_durations_ms": [t * 1000 for t in ticks[:8]],
                   "first_delta_z": deltas[:8], "first_climb_tick": first_climb}
            schedules.append(row)
            checks.check("listen_schedule", first_climb is not None if mode == "paired" else
                         (first_climb is not None) == (render_hz in (72, 77)), row)
    results["listen_schedules"] = schedules

    contacts = []
    for angle in (0, 5, 15, 30, 45, 46, 60, 75, 85, 90, 120, 150, 180):
        a = math.radians(angle)
        n = vec((-math.sin(a), 0, math.cos(a)))
        tangent = vec((math.cos(a), 0, math.sin(a)))
        world = World(halfspace(n, name=f"plane_{angle}"))
        for hz in RATES:
            for direction in (-1, 1):
                initial = tangent * F(120 * direction) - n * F(100)
                for mode in ("split", "paired"):
                    ent = Entity(n * F(EPS), initial.copy())
                    tick(ent, 1 / hz, world, mode, thud_touch)
                    expected = initial.copy()
                    expected[2] = F(float(expected[2]) - 800 / hz)
                    expected = clip(expected, n)
                    row = {"angle": angle, "hz": hz, "direction": direction, "mode": mode,
                           "normal_velocity": float(dot(ent.vel, n)),
                           "projection_error": float(np.max(np.abs(ent.vel - expected))),
                           "penetration": world.penetration(ent.pos)}
                    contacts.append(row)
                    if mode == "paired":
                        checks.check("single_plane", row["projection_error"] < 0.001 and
                                     row["penetration"] < 0.001, row)
    results["contacts"] = contacts

    # Slide without friction or input, actively moving each tick. This isolates
    # gravity integration, rather than emulating a walking monster's AI.
    slopes = []
    for angle in (5, 15, 30, 45, 46, 60, 75, 85):
        a = math.radians(angle)
        n = vec((-math.sin(a), 0, math.cos(a)))
        tangent = vec((-math.cos(a), 0, -math.sin(a)))
        world = World(halfspace(n, name="slope"))
        accel = 800 * math.sin(a)
        for hz in RATES:
            for mode in ("split", "paired"):
                start = n * F(EPS)
                ent = Entity(start.copy(), tangent * F(100))
                depth, normal_speed = 0.0, 0.0
                for _ in range(hz * 2):
                    ent.ground = False
                    tick(ent, 1 / hz, world, mode)
                    depth = max(depth, world.penetration(ent.pos))
                    normal_speed = max(normal_speed, abs(float(dot(ent.vel, n))))
                travel = float(dot(ent.pos - start, tangent))
                speed = float(dot(ent.vel, tangent))
                expected_travel = 200 + 0.5 * accel * (4 + 2 * H)
                expected_speed = 100 + 2 * accel
                row = {"angle": angle, "hz": hz, "mode": mode, "travel": travel,
                       "speed": speed, "travel_error": travel - expected_travel,
                       "speed_error": speed - expected_speed, "penetration": depth,
                       "max_normal_speed": normal_speed}
                slopes.append(row)
                if mode == "paired":
                    checks.check("long_slope", depth < 0.001 and abs(row["speed_error"]) < 0.1
                                 and abs(row["travel_error"]) < 0.1, row)
    results["slopes"] = slopes

    # Finite geometry exercises changes of support plane and player stair
    # retries. Differences vs 72Hz are measured, not assumed to be errors.
    top = 64 * math.tan(math.radians(30))
    scenes = {
        "three_8_unit_steps": World(floor,
            box((0, -100, -100), (24, 100, 8), "step1"),
            box((24, -100, -100), (48, 100, 16), "step2"),
            box((48, -100, -100), (1000, 100, 24), "step3")),
        "20_unit_obstacle": World(floor, box((0, -100, -100), (1000, 100, 20), "too_high")),
        "ramp_to_platform": World(floor, ramp(0, 64, 0, 30, "ramp"),
            box((64, -100, -100), (1000, 100, top), "platform")),
        "ramp_to_steep_slope": World(floor, ramp(0, 64, 0, 30, "ramp"),
            ramp(64, 200, top, 60, "steep")),
        "low_ceiling_stair": World(floor,
            box((0, -100, -100), (1000, 100, 8), "step"),
            halfspace((0, 0, -1), -12, "ceiling")),
    }
    players = []
    for name, world in scenes.items():
        for hz in RATES:
            for mode in ("split", "paired"):
                ent = Entity(vec((-10, 0, EPS)), vec((100, 0, 0)), ground=True)
                depth = 0.0
                for _ in range(hz):
                    ent.vel[0] = F(100)
                    player_tick(ent, 1 / hz, world, mode)
                    depth = max(depth, world.penetration(ent.pos))
                row = {"scene": name, "hz": hz, "mode": mode, "pos": ent.pos,
                       "vel": ent.vel, "penetration": depth}
                players.append(row)
                checks.check("player_geometry", depth < 0.001 and np.isfinite(ent.vel).all(), row)
                if mode == "paired" and hz <= 72 and name == "three_8_unit_steps":
                    checks.check("player_stairs", ent.pos[0] > 48 and abs(ent.pos[2] - 24 - EPS) < 0.001, row)
                if mode == "paired" and name == "20_unit_obstacle":
                    checks.check("player_tall_step", ent.pos[0] <= 0 and ent.pos[2] < 1, row)
    results["players"] = players
    for name in scenes:
        rows = [r for r in players if r["scene"] == name and r["hz"] == 72]
        checks.check("player_72hz_parity", np.array_equal(rows[0]["pos"], rows[1]["pos"])
                     and np.array_equal(rows[0]["vel"], rows[1]["vel"]), rows)

    # Probe where sweep and end velocities have opposite vertical signs.
    # Broad random speeds rarely exercise this narrow region around the apex.
    apex = []
    world = World(floor, halfspace((-1, 0, 0), -8, "wall_x"),
                  halfspace((0, -1, 0), -7, "wall_y"),
                  halfspace((0, 0, -1), -4, "ceiling"))
    for hz in RATES:
        dt = 1 / hz
        for position in ((8 - EPS, 7 - EPS, EPS), (8 - EPS, 7 - EPS, 4 - EPS)):
            for center in (800 * dt, 400 * (dt + H)):
                for offset in (-1, -0.1, -0.001, 0, 0.001, 0.1, 1):
                    initial = vec((100, 80, center + offset))
                    ent = Entity(vec(position), initial.copy())
                    for _ in range(4):
                        ent.ground = False
                        tick(ent, dt, world, "paired")
                        checks.check("apex_corners", world.penetration(ent.pos) < 0.001
                                     and np.isfinite(ent.vel).all(),
                                     {"hz": hz, "initial": initial, "pos": ent.pos, "vel": ent.vel})
                    apex.append({"hz": hz, "initial": initial, "pos": ent.pos, "vel": ent.vel})
    results["apex"] = apex

    # Exact port-to-port regression comparisons at 72Hz and with analytic
    # physics disabled. Off-rate analytic results need not equal the old bug.
    rng = np.random.default_rng(1027)
    parity_errors = []
    geometry_errors = []
    for case in range(random_cases):
        angle = rng.uniform(0, 80)
        a = math.radians(angle)
        n = vec((-math.sin(a), 0, math.cos(a)))
        world = World(halfspace(n, name="ramp"), halfspace((-1, 0, 0), -8, "wall_x"),
                      halfspace((0, -1, 0), -7, "wall_y"), halfspace((0, 0, -1), -24, "ceiling"))
        p = vec((rng.uniform(-5, 0), rng.uniform(-5, 0), rng.uniform(0.1, 15)))
        velocity = vec(rng.uniform(-600, 600, 3))
        callback_kind = case % 4

        def cb(ent, trace):
            if callback_kind == 1:
                thud_touch(ent, trace)
            elif callback_kind == 2:
                ent.vel = vec((90, -40, 270))
            elif callback_kind == 3:
                ent.free = True

        for analytic, dt in ((True, H), (False, 1 / RATES[case % len(RATES)])):
            old = Entity(p.copy(), velocity.copy())
            new = old.clone()
            old_blocked = tick(old, dt, world, "split", cb, analytic=analytic)
            new_blocked = tick(new, dt, world, "paired", cb, analytic=analytic)
            equal = (np.array_equal(old.pos, new.pos) and np.array_equal(old.vel, new.vel)
                     and old.ground == new.ground and old.free == new.free and old.thud == new.thud
                     and old_blocked == new_blocked and len(old.touches) == len(new.touches)
                     and all(a[0] == b[0] and np.array_equal(a[1], b[1])
                             for a, b in zip(old.touches, new.touches)))
            detail = {"case": case, "analytic": analytic, "dt": dt, "angle": angle,
                      "old_pos": old.pos, "new_pos": new.pos, "old_vel": old.vel, "new_vel": new.vel}
            checks.check("legacy_parity", equal, detail)
            if not equal and len(parity_errors) < 5:
                parity_errors.append(detail)
        # Off-rate motion through the same geometry, without arbitrary callbacks.
        ent = Entity(p.copy(), velocity.copy())
        dt = 1 / RATES[case % len(RATES)]
        max_depth = 0.0
        for _ in range(4):
            ent.ground = False
            initial_end = ent.vel.copy()
            initial_end[2] = F(float(initial_end[2]) - ent.gravity * dt)
            tick(ent, dt, world, "paired")
            max_depth = max(max_depth, world.penetration(ent.pos))
            checks.check("collision_energy", np.linalg.norm(ent.vel.astype(float)) <=
                         np.linalg.norm(initial_end.astype(float)) + 0.001,
                         {"case": case, "before": initial_end, "after": ent.vel})
        okay = max_depth < 0.001 and np.isfinite(ent.vel).all() and np.isfinite(ent.pos).all()
        detail = {"case": case, "angle": angle, "dt": dt, "depth": max_depth,
                  "pos": ent.pos, "vel": ent.vel}
        checks.check("random_geometry", okay, detail)
        if not okay and len(geometry_errors) < 5:
            geometry_errors.append(detail)
    results["parity_examples"] = parity_errors
    results["geometry_examples"] = geometry_errors

    callbacks = []
    for hz in RATES:
        for mode in ("split", "paired"):
            def launch(ent):
                ent.vel = vec((80, 0, 270))
                ent.ground = False

            ent = Entity(vec((0, 0, 10)), vec((100, 0, -5)))
            tick(ent, 1 / hz, empty, mode, link_callback=launch)
            row = {"hz": hz, "mode": mode, "post_trigger_vz": float(ent.vel[2])}
            callbacks.append(row)
            if mode == "paired":
                checks.check("trigger_velocity", ent.vel[2] == 270, row)
    results["callbacks"] = callbacks

    bounces = []
    for angle in (0, 15, 30, 45, 60):
        a = math.radians(angle)
        n = vec((-math.sin(a), 0, math.cos(a)))
        world = World(halfspace(n, name="bounce_plane"))
        for hz in RATES:
            old = Entity(n * F(EPS), vec((100, 0, -300)))
            new = old.clone()
            toss_tick(old, 1 / hz, world, "split")
            toss_tick(new, 1 / hz, world, "paired")
            difference = float(np.max(np.abs(old.vel - new.vel)))
            row = {"angle": angle, "hz": hz, "velocity_difference": difference,
                   "normal_velocity": float(dot(new.vel, n))}
            checks.check("bounce", difference < 0.0001 and row["normal_velocity"] >= -0.0001, row)
            bounces.append(row)
    results["bounces"] = bounces
    return results


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for mode, label, color in (("split", "Previous split gravity", "#bc4b3a"),
                               ("paired", "Paired velocities", "#176b8e")):
        rows = [r for r in results["slopes"] if r["mode"] == mode and r["hz"] == 60]
        axes[0].plot([r["angle"] for r in rows], [r["speed_error"] for r in rows],
                     "o-", label=label, color=color)
        frogs = [r for r in results["frogbot"] if r["mode"] == mode and r["height"] == 8 and r["hz"] <= 72]
        axes[1].plot([r["hz"] for r in frogs], [r["delta_z"] for r in frogs],
                     "o-", label=label, color=color)
    axes[0].set(xlabel="Slope angle (degrees)", ylabel="Speed error after 2 s (units/s)",
                title="Frictionless slope at 60 physics ticks/s\nError vs projected 72 Hz freefall formula")
    axes[1].axhspan(-0.1, 0.1, color="#83a86d", alpha=0.25, label="Frogbot acceptance range")
    axes[1].set(xlabel="Physics tick rate (Hz), not rendering FPS", ylabel="Pre/post vertical velocity difference (units/s)",
                title="Floor contact before an 8-unit step\nFrogbot-style callback clears the ground flag")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(out / "slopes.png", dpi=160)
    plt.close(fig)


def write_report(results, out):
    failures = sum(results["failure_counts"].values())
    total = sum(results["checks"].values())
    slopes = results["slopes"]
    paired = [r for r in slopes if r["mode"] == "paired"]
    lines = [
        "# Gravity and collision simulation", "",
        f"{total:,} numerical/model checks; {failures} failures.", "",
        "Python ports of the previous and proposed movement paths, using float32 arithmetic.",
        "Geometry consists of convex solid brushes representing expanded hulls. No engine compilation or execution.",
        "This does not validate the real BSP tracer, QuakeC VM, moving pushers, water, or compiler/FMA behavior.", "",
        "Primary physics rates: 30, 40, 58, 60, and 72 Hz. Rates of 77, 120, 144, and 240 Hz are additional stress tests only.",
        "Rendering above 72 FPS is decoupled from physics; those stress rates are not normal vkQuake physics rates.", "",
        f"Baseline HEAD: `{results['metadata']['head']}`.",
        f"Candidate sv_phys.c SHA-256: `{results['metadata']['candidate_source_sha256']}`.", "",
        "## Frogbot reproduction", "",
        "At 60 physics ticks/s, the old path produces delta_velocity_z = +1.111111 after the touch callback clears FL_ONGROUND.",
        "The paired path produces zero and attempts walkmove. The model climbs 1-, 8-, 16-, and 18-unit steps and rejects a 20-unit step.",
        "The simulated recovery is the relevant obstruction gate and downward walkmove sweep, not the whole bot AI.", "",
        "In the extra 240-Hz PHYSICS stress test, both paths initially miss the floor: downward travel from rest (0.03009 units) is less than",
        "the 0.03125-unit clearance. This is an uncompleted landing, not the reported post-collision velocity error.",
        "These are PHYSICS rates; rendering at 120 or 240 FPS does not imply those physics rates.", "",
        "## Ideal listen-server schedules", "",
        "A second model ports host.c's actual accumulator at steady rendering FPS (no rendering jitter).",
        "It reproduces the reported 77-FPS exception as well as the failures at 60 and 120 rendering FPS.", "",
        "| Rendering FPS | Physics ticks in 1 s | Old first successful stair tick | Paired first successful stair tick |",
        "|---|---:|---:|---:|",
    ]
    for fps in (60, 72, 77, 120, 240):
        old = next(r for r in results["listen_schedules"] if r["render_fps"] == fps and r["mode"] == "split")
        new = next(r for r in results["listen_schedules"] if r["render_fps"] == fps and r["mode"] == "paired")
        lines.append(f"| {fps} | {old['physics_ticks_in_one_second']} | {old['first_climb_tick'] or 'none'} | {new['first_climb_tick']} |")
    lines += ["", "At 77 rendering FPS, periodic ticks are close enough to 1/72 that the old gravity residual falls below the bot's 0.1 threshold.",
        "At 120 and 240 rendering FPS, this ideal listen-server model runs at 60 physics ticks/s, retaining the old +1.111111 obstruction delta.", "",
        "## Slope results", "",
        "Frictionless motion actively simulated every tick. The reference is the gravity component projected onto an infinite plane,",
        "integrated with the 72 Hz displacement formula. This is not a full player movement benchmark.", "",
        "| Slope | Old speed error at 60 Hz | Paired speed error at 60 Hz |",
        "|---|---:|---:|",
    ]
    for angle in (5, 15, 30, 45, 46, 60, 75, 85):
        old = next(r for r in slopes if r["angle"] == angle and r["hz"] == 60 and r["mode"] == "split")
        new = next(r for r in slopes if r["angle"] == angle and r["hz"] == 60 and r["mode"] == "paired")
        lines.append(f"| {angle} degrees | {old['speed_error']:+.6f} | {new['speed_error']:+.6f} |")
    lines += ["", "Errors are in units/s after two seconds.", "",
              f"Across the slope/rate sweep, paired maximum speed error: {max(abs(r['speed_error']) for r in paired):.6f} units/s;",
              f"maximum travel error: {max(abs(r['travel_error']) for r in paired):.6f} units.", "",
              "The old ground-flag shortcut discards the remaining tangent gravity on walkable slopes. On steeper slopes it can",
              "reintroduce velocity into the plane. The paired model clips both contributions.", "",
              "## Regression coverage", ""]
    for key, count in results["checks"].items():
        lines.append(f"- {key}: {count:,} checks, {results['failure_counts'].get(key, 0)} failures.")
    lines += ["", "Legacy parity compares the independent old/new Python mover ports exactly at 72 Hz and with analytic physics disabled.",
              "Random geometry checks finite state and nonpenetration; energy checks reject speed amplification beyond the gravity-updated input.",
              "They do not prove identical off-rate trajectories. Tests include impact callbacks clearing flags, replacing velocity, and removing entities.",
              "Post-movement trigger tests verify that a replacement velocity of 270 units/s remains unchanged.", "",
              "The finite scenes cover 8-unit staircases, a 20-unit obstacle, a low ceiling, ramp/platform transitions, and ramp/steep-slope transitions.",
              "Player simulation uses pr_checkextension=1 (no TryUnstick), no water/pushers, a fixed yaw, and prescribed horizontal input.", "",
              "## Remaining timestep dependence", "",
              "For the prescribed-input ramp-to-platform scene, final x after one second is:", "",
              "| Physics rate | Old | Paired |", "|---|---:|---:|"]
    for hz in (30, 40, 60, 72):
        old = next(r for r in results["players"] if r["scene"] == "ramp_to_platform" and r["hz"] == hz and r["mode"] == "split")
        new = next(r for r in results["players"] if r["scene"] == "ramp_to_platform" and r["hz"] == hz and r["mode"] == "paired")
        lines.append(f"| {hz} | {old['pos'][0]:.6f} | {new['pos'][0]:.6f} |")
    lines += ["", "The patch is not exact fixed-72-Hz gameplay: input and collision timing still depend on the tick size.",
              "In this transition scene the paired endpoint at 60 Hz is slightly farther from the 72 Hz endpoint than the old path.",
              "Do not interpret the freefall proof as a proof of universal collision or mod compatibility.", "",
              "![Simulation results](slopes.png)", ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=root / "build/physics-gravity-sim")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = Checks()
    results = run(checks, args.cases)
    source = (root / "Quake/sv_phys.c").read_bytes()
    results["metadata"] = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "candidate_source_sha256": hashlib.sha256(source).hexdigest(),
        "seed": 1027, "random_cases": args.cases, "tick_rates": RATES,
        "limits": "Python source ports; convex brush tracing, not engine BSP/QC/pushers. No engine compilation or execution.",
    }
    results["checks"] = dict(checks.counts)
    results["failures"] = checks.failures
    results["failure_counts"] = dict(checks.failure_counts)
    results["branch_counts"] = dict(COUNTS)
    (args.output / "results.json").write_text(json.dumps(results, default=plain, indent=2) + "\n")
    plot(results, args.output)
    write_report(results, args.output)
    print(json.dumps({k: results[k] for k in ("metadata", "checks", "branch_counts", "failure_counts", "failures")},
                     default=plain, indent=2))
    return bool(checks.failures)


if __name__ == "__main__":
    raise SystemExit(main())
