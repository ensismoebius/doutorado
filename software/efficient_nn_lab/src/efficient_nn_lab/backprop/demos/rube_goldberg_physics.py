"""The actual mechanics of the Rube Goldberg machine -- a real pymunk 2D
physics simulation, re-run whenever the network's free parameters change
(see ``simulate_machine`` below).

Why a real physics engine, not hand-authored motion: an earlier version of
this demo faked the ball's path with a sine-bob linear interpolation
between five fixed station slots. The user asked for the ball to roll down
the ramps along their actual slope, the lever to physically tip under
impact, and the pulley to visibly spin -- none of which a hand-tuned tween
can do convincingly, because it never resolves a contact force.

The LEVER and the ROLDANA (pulley) are the machine's two draggable handles,
and they map onto the only two genuinely free numbers in this 1->1->1
network: dragging the lever sets ``w2``, dragging the pulley sets the input
``x`` (see ``rube_goldberg_chain.py``). Everything else -- both ramps'
steepness, the funnel's drop -- is a CONSEQUENCE of those two numbers
through the real forward/backward pass
(``chain_rule_layers.compute_chain_1_1_1``), never something set directly:
ramp 1's angle tracks ``sigma'(z2)``, ramp 2's tracks ``sigma'(z1)``, both
always true derivatives of sigmoid and therefore always in ``[0, 0.25]`` no
matter what ``w2``/``x`` are dragged to -- which is also why the geometry
below never needs an escape hatch for an out-of-range slope: the real math
already keeps it bounded.

Still deterministic (ESPECIFICACAO_DLVL.md #35): for any FIXED ``(sp2,
sp1, w2)`` the simulation has no randomness anywhere (fixed initial
position/velocity, fixed gravity, fixed timestep), so re-running it for the
same inputs always produces the bit-identical trajectory -- verified by
``tests/test_backprop.py``'s determinism check. It no longer runs once at
import time, though: it now depends on the demo's ``w2``/``x`` parameters,
so it is re-run (still with zero randomness) every time those change,
exactly like every other demo's ``_build_frames`` already re-runs its own
numeric derivation on every ``set_parameter`` call.

Finding a physical layout that doesn't strand the ball -- across the WHOLE
range the user can now drag the ramps' steepness and the lever's tilt to,
not just one fixed case -- took three failed geometries for the original
fixed-slope version (see git history if curious): the first let the ball
overshoot every wall because nothing bled off its energy; the second
stalled it dead on flat connector floors because ``space.damping`` (added
to fix the first problem) kills horizontal speed on any stretch gravity
isn't actively pulling it along. The fix for both, kept here: every
connector is part of ONE continuous, strictly-descending slope from the
funnel to the bucket (so gravity always has a forward component), and the
pulley is a circle embedded in that same slope rather than a separate flat
"bench" the ball could die on. ``test_rube_goldberg_physics_robust_across_drag_range``
sweeps the full ``w2``/``x`` drag bounds and asserts the ball always
reaches the bucket, since a layout that only worked for the one default
case would silently strand the ball the first time someone actually drags
something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pymunk

GRAVITY = -10.0
DT = 1.0 / 120.0
MAX_STEPS = 1600
SPACE_DAMPING = 0.992  # mild rolling-resistance/drag so the ball settles in
# the bucket instead of bouncing forever (combined with zero elasticity
# everywhere -- a Rube Goldberg ball should roll, not billiard around).

BALL_RADIUS = 0.28
PULLEY_RADIUS = 0.45

#: Drag bounds for the two draggable handles -- exported so the demo
#: module (slider ranges) and the canvas drag handler (gesture-to-value
#: mapping, clamping) share the exact same numbers instead of each
#: guessing its own.
W2_DRAG_BOUNDS = (-3.0, 3.0)
X_DRAG_BOUNDS = (-2.0, 2.0)

#: A sigmoid derivative is always in [0, 0.25] (max at a=0.5) -- true for
#: ANY w2/x, which is what makes the ramp-angle mapping below safe without
#: ever needing to clamp an out-of-range input.
_SIGMOID_DERIV_MAX = 0.25
_RAMP_MIN_ANGLE_DEG = 10.0
_RAMP_MAX_ANGLE_DEG = 38.0
_LEVER_MIN_TILT_DEG = 6.0
_LEVER_MAX_TILT_DEG = 26.0

# Fixed waypoints/runs that never change with w2/x -- only the two ramps'
# drop and the lever's tilt are parametric; everything downstream of them
# keeps the same run/drop it always had, so the rest of the canvas layout
# (pulley, bucket, caption rows) never has to move either.
P_SPAWN = (2.6, 7.00)
P_FUNNEL_TOP = (2.6, 6.60)
_RAMP1_X_RUN = 3.00
_CONNECTOR1_DX, _CONNECTOR1_DY = 0.65, -0.30
LEVER_HALF_LEN = 1.35
_CONNECTOR2_DX, _CONNECTOR2_DY = 0.30, -0.10
_RAMP2_X_RUN = 3.15
_FLOOR_DX, _FLOOR_DY = 3.05, -0.90
_BUCKET_DROP_DX, _BUCKET_DROP_DY = 0.75, -1.00
_BUCKET_FLOOR_DX = 0.70
_BUCKET_WALL_HEIGHT = 2.30


def _ramp_angle_rad(sigmoid_deriv: float) -> float:
    t = max(0.0, min(1.0, sigmoid_deriv / _SIGMOID_DERIV_MAX))
    return math.radians(_RAMP_MIN_ANGLE_DEG + t * (_RAMP_MAX_ANGLE_DEG - _RAMP_MIN_ANGLE_DEG))


def _lever_tilt_rad(w2: float) -> float:
    lo, hi = W2_DRAG_BOUNDS
    t = max(0.0, min(1.0, abs(w2) / max(abs(lo), abs(hi))))
    return math.radians(_LEVER_MIN_TILT_DEG + t * (_LEVER_MAX_TILT_DEG - _LEVER_MIN_TILT_DEG))


@dataclass(frozen=True)
class PhysicsSample:
    t: float
    ball_x: float
    ball_y: float
    ball_angle: float
    lever_angle: float
    pulley_angle: float


@dataclass(frozen=True)
class MachineLayout:
    """Every waypoint/body the renderer draws, for ONE choice of (sp2, sp1,
    w2) -- the single source of truth both `_build_space` and the renderer
    read from, so the picture can never disagree with what the ball
    actually collided with."""

    funnel_top: tuple[float, float]
    ramp1_top: tuple[float, float]
    ramp1_bottom: tuple[float, float]
    lever_left: tuple[float, float]
    lever_right: tuple[float, float]
    lever_pivot: tuple[float, float]
    lever_rest_angle: float
    ramp2_top: tuple[float, float]
    ramp2_bottom: tuple[float, float]
    bucket_drop_top: tuple[float, float]
    bucket_floor_left: tuple[float, float]
    bucket_right_wall_x: float
    bucket_wall_top: float
    pulley_center: tuple[float, float]


def _build_layout(sp2: float, sp1: float, w2: float) -> MachineLayout:
    funnel_top = P_FUNNEL_TOP
    ramp1_top = funnel_top
    angle1 = _ramp_angle_rad(sp2)
    ramp1_bottom = (ramp1_top[0] + _RAMP1_X_RUN, ramp1_top[1] - _RAMP1_X_RUN * math.tan(angle1))

    lever_left = (ramp1_bottom[0] + _CONNECTOR1_DX, ramp1_bottom[1] + _CONNECTOR1_DY)
    tilt = _lever_tilt_rad(w2)
    rest_angle = -tilt  # always right-lower (ball keeps moving forward downhill
    # regardless of w2's sign -- only the ball's COLOR encodes the sign, same
    # reasoning as the fixed-geometry version this replaces).
    span = 2.0 * LEVER_HALF_LEN
    lever_right = (lever_left[0] + span * math.cos(rest_angle), lever_left[1] + span * math.sin(rest_angle))
    lever_pivot = ((lever_left[0] + lever_right[0]) / 2.0, (lever_left[1] + lever_right[1]) / 2.0)

    ramp2_top = (lever_right[0] + _CONNECTOR2_DX, lever_right[1] + _CONNECTOR2_DY)
    angle2 = _ramp_angle_rad(sp1)
    ramp2_bottom = (ramp2_top[0] + _RAMP2_X_RUN, ramp2_top[1] - _RAMP2_X_RUN * math.tan(angle2))

    bucket_drop_top = (ramp2_bottom[0] + _FLOOR_DX, ramp2_bottom[1] + _FLOOR_DY)
    bucket_floor_left = (bucket_drop_top[0] + _BUCKET_DROP_DX, bucket_drop_top[1] + _BUCKET_DROP_DY)
    bucket_right_wall_x = bucket_floor_left[0] + _BUCKET_FLOOR_DX
    bucket_wall_top = bucket_floor_left[1] + _BUCKET_WALL_HEIGHT

    floor_t = 0.5  # pulley sits at the floor segment's midpoint
    pulley_x = ramp2_bottom[0] + floor_t * (bucket_drop_top[0] - ramp2_bottom[0])
    pulley_y = ramp2_bottom[1] + floor_t * (bucket_drop_top[1] - ramp2_bottom[1])

    return MachineLayout(
        funnel_top=funnel_top, ramp1_top=ramp1_top, ramp1_bottom=ramp1_bottom,
        lever_left=lever_left, lever_right=lever_right, lever_pivot=lever_pivot,
        lever_rest_angle=rest_angle, ramp2_top=ramp2_top, ramp2_bottom=ramp2_bottom,
        bucket_drop_top=bucket_drop_top, bucket_floor_left=bucket_floor_left,
        bucket_right_wall_x=bucket_right_wall_x, bucket_wall_top=bucket_wall_top,
        pulley_center=(pulley_x, pulley_y),
    )


def _build_space(layout: MachineLayout) -> tuple[pymunk.Space, pymunk.Body, pymunk.Body, pymunk.Body]:
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    space.damping = SPACE_DAMPING

    def seg(a: tuple[float, float], b: tuple[float, float], friction: float = 0.95) -> None:
        shape = pymunk.Segment(space.static_body, a, b, 0.03)
        shape.friction = friction
        shape.elasticity = 0.0
        space.add(shape)

    seg(layout.funnel_top, layout.ramp1_top)
    seg(layout.ramp1_top, layout.ramp1_bottom)
    seg(layout.ramp1_bottom, layout.lever_left)
    seg(layout.lever_right, layout.ramp2_top)
    seg(layout.ramp2_top, layout.ramp2_bottom)
    seg(layout.ramp2_bottom, layout.bucket_drop_top)
    seg(layout.bucket_drop_top, layout.bucket_floor_left)
    seg(layout.bucket_floor_left, (layout.bucket_right_wall_x, layout.bucket_floor_left[1]))
    seg((layout.bucket_right_wall_x, layout.bucket_floor_left[1]), (layout.bucket_right_wall_x, layout.bucket_wall_top))

    lever_mass = 1.0
    lever_moment = pymunk.moment_for_segment(lever_mass, (-LEVER_HALF_LEN, 0), (LEVER_HALF_LEN, 0), 0.08)
    lever_body = pymunk.Body(lever_mass, lever_moment)
    lever_body.position = layout.lever_pivot
    lever_body.angle = layout.lever_rest_angle
    lever_shape = pymunk.Segment(lever_body, (-LEVER_HALF_LEN, 0), (LEVER_HALF_LEN, 0), 0.08)
    lever_shape.friction = 0.95
    lever_shape.elasticity = 0.0
    space.add(lever_body, lever_shape)
    pivot_joint = pymunk.PivotJoint(space.static_body, lever_body, layout.lever_pivot)
    pivot_joint.collide_bodies = False
    # DampedRotarySpring's rest_angle is body_a.angle - body_b.angle; body_a
    # is the static anchor (always angle 0), so the lever's OWN equilibrium
    # angle is the negative of what's passed in here -- flip the sign so it
    # actually settles at layout.lever_rest_angle (confirmed empirically
    # against a standalone trace, not derived from the pymunk docs alone).
    spring = pymunk.DampedRotarySpring(
        space.static_body, lever_body, rest_angle=-layout.lever_rest_angle, stiffness=35.0, damping=9.0
    )
    space.add(pivot_joint, spring)

    pulley_mass = 0.5
    pulley_moment = pymunk.moment_for_circle(pulley_mass, 0, PULLEY_RADIUS)
    pulley_body = pymunk.Body(pulley_mass, pulley_moment)
    pulley_body.position = layout.pulley_center
    pulley_shape = pymunk.Circle(pulley_body, PULLEY_RADIUS)
    pulley_shape.friction = 1.6
    pulley_shape.elasticity = 0.0
    space.add(pulley_body, pulley_shape)
    pulley_pivot = pymunk.PivotJoint(space.static_body, pulley_body, layout.pulley_center)
    pulley_pivot.collide_bodies = False
    space.add(pulley_pivot)

    ball_mass = 0.5
    ball_moment = pymunk.moment_for_circle(ball_mass, 0, BALL_RADIUS)
    ball_body = pymunk.Body(ball_mass, ball_moment)
    ball_body.position = P_SPAWN
    # A tiny rightward nudge breaks the otherwise perfectly symmetric drop
    # straight onto the funnel chute's corner, which has no preferred
    # direction to roll and would just balance there forever.
    ball_body.velocity = (0.4, 0)
    ball_shape = pymunk.Circle(ball_body, BALL_RADIUS)
    ball_shape.friction = 0.9
    ball_shape.elasticity = 0.0
    space.add(ball_body, ball_shape)

    return space, ball_body, lever_body, pulley_body


@dataclass(frozen=True)
class MachineRun:
    """One full simulation: the layout it used, the trajectory it produced,
    and the six narrative-leg x-bounds for THIS layout (the ramps' x-span
    is fixed, but the lever's span shifts slightly with its tilt, so the
    bounds are computed per-run rather than being module constants)."""

    layout: MachineLayout
    trajectory: list[PhysicsSample] = field(repr=False)
    leg_x_bounds: tuple[tuple[float, float], ...]


def simulate_machine(sp2: float, sp1: float, w2: float) -> MachineRun:
    """Run the machine once for this choice of (sp2, sp1, w2).

    ``sp2``/``sp1`` set the two ramps' steepness (always in [0, 0.25], the
    sigmoid derivative's own range); ``w2`` sets the lever's tilt magnitude
    (sign is irrelevant to the physics -- see module docstring). Called
    from ``rube_goldberg_chain.py`` once per ``_build_frames()``, i.e. once
    per slider/drag change, same as every other demo's numeric derivation.
    """
    layout = _build_layout(sp2, sp1, w2)
    space, ball, lever, pulley = _build_space(layout)
    samples: list[PhysicsSample] = []
    rest_streak = 0
    for i in range(MAX_STEPS):
        space.step(DT)
        samples.append(PhysicsSample(i * DT, ball.position.x, ball.position.y, ball.angle, lever.angle, pulley.angle))
        at_rest = abs(ball.velocity.x) < 0.05 and abs(ball.velocity.y) < 0.05
        if ball.position.x > layout.bucket_right_wall_x - 1.0 and at_rest:
            rest_streak += 1
            if rest_streak > 30:
                break
        else:
            rest_streak = 0

    leg_x_bounds = (
        (P_SPAWN[0], layout.ramp1_top[0]),
        (layout.ramp1_top[0], layout.ramp1_bottom[0]),
        (layout.ramp1_bottom[0], layout.lever_right[0]),
        (layout.lever_right[0], layout.ramp2_bottom[0]),
        (layout.ramp2_bottom[0], layout.bucket_drop_top[0]),
        (layout.bucket_drop_top[0], layout.bucket_right_wall_x),
    )
    return MachineRun(layout=layout, trajectory=samples, leg_x_bounds=leg_x_bounds)


def _index_at_or_after_x(run: MachineRun, x_threshold: float) -> int:
    for i, s in enumerate(run.trajectory):
        if s.ball_x >= x_threshold:
            return i
    return len(run.trajectory) - 1


def leg_samples(run: MachineRun, leg_index: int, n: int) -> list[PhysicsSample]:
    """``n`` evenly-spaced physics samples spanning narrative leg ``leg_index``
    of ``run`` -- used to populate a checkpoint-to-checkpoint tween with
    real simulated motion instead of a linear/eased interpolation."""
    x0, x1 = run.leg_x_bounds[leg_index]
    start = _index_at_or_after_x(run, x0)
    end = _index_at_or_after_x(run, x1)
    if end <= start:
        end = min(start + 1, len(run.trajectory) - 1)
    return [run.trajectory[round(start + (end - start) * k / max(n - 1, 1))] for k in range(n)]


def leg_boundary_sample(run: MachineRun, leg_index: int, end: bool) -> PhysicsSample:
    """The physics sample at the start (``end=False``) or end (``end=True``)
    of narrative leg ``leg_index`` of ``run`` -- what a checkpoint's own
    (non-tween) frame shows, so it lines up with the tween that arrives at it."""
    x0, x1 = run.leg_x_bounds[leg_index]
    return run.trajectory[_index_at_or_after_x(run, x1 if end else x0)]


def final_sample(run: MachineRun) -> PhysicsSample:
    """Where the ball actually comes to rest -- the bucket's picture uses
    this instead of a hand-picked position, so it can never disagree with
    where the simulated ball really lands."""
    return run.trajectory[-1]
