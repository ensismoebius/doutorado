"""The actual mechanics of the Rube Goldberg machine -- a real pymunk 2D
physics simulation, run once at import time (see module-level
``TRAJECTORY`` below).

Why a real physics engine, not hand-authored motion: the first version of
this demo faked the ball's path with a sine-bob linear interpolation
between five fixed station slots. The user explicitly asked for the ball to
roll down the ramps along their actual slope, the lever to physically tip
under impact, and the pulley to visibly spin -- none of which a hand-tuned
tween can do convincingly, because it never resolves a contact force.

Still deterministic (ESPECIFICACAO_DLVL.md #35): the simulation has no
randomness anywhere (fixed initial position/velocity, fixed gravity, fixed
timestep), so re-running it always produces the bit-identical trajectory --
verified by ``tests/test_backprop.py``'s determinism check. It is ALSO
completely independent of the demo's ``target`` slider: the physics is pure
mechanics (ramps, a pivoted beam, a free-spinning disc, gravity), and never
reads a single one of the five chain-rule factors. Those numbers only
decide what's PRINTED on the chips the ball passes under -- never where the
ball goes. That is why this module computes ``TRAJECTORY`` exactly once,
instead of re-simulating every time a slider moves.

Finding the physical layout that doesn't strand the ball took three failed
geometries (see git history if curious): the first let the ball overshoot
every wall because nothing bled off its energy; the second stalled it dead
on flat connector floors because ``space.damping`` (added to fix the first
problem) kills horizontal speed on any stretch gravity isn't actively
pulling it along. The fix for both: every connector is part of ONE
continuous, strictly-descending slope from the funnel to the bucket (so
gravity always has a forward component), and the pulley is a circle
embedded in that same slope rather than a separate flat "bench" the ball
could die on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pymunk

GRAVITY = -10.0
DT = 1.0 / 120.0
MAX_STEPS = 1600
SPACE_DAMPING = 0.992  # mild rolling-resistance/drag so the ball settles in
# the bucket instead of bouncing forever (combined with zero elasticity
# everywhere -- a Rube Goldberg ball should roll, not billiard around).

BALL_RADIUS = 0.28

#: The machine's skeleton, as one continuous, strictly-descending chain of
#: waypoints -- the renderer draws the static track from these same points,
#: so the drawing can never disagree with what the ball actually collides
#: with.
P_SPAWN = (2.6, 7.00)
P_FUNNEL_TOP = (2.6, 6.60)
P_RAMP1_TOP = (4.00, 5.80)
P_RAMP1_BOTTOM = (7.00, 4.30)
P_LEVER_LEFT = (7.65, 4.00)
P_LEVER_RIGHT = (10.35, 3.55)
P_RAMP2_TOP = (10.65, 3.45)
P_RAMP2_BOTTOM = (13.80, 2.20)
P_FLOOR_START = P_RAMP2_BOTTOM
P_BUCKET_DROP_TOP = (16.85, 1.30)
P_BUCKET_FLOOR_LEFT = (17.60, 0.30)
BUCKET_RIGHT_WALL_X = 18.30
BUCKET_WALL_TOP = 2.60

LEVER_PIVOT = (9.0, 3.775)
LEVER_HALF_LEN = 1.35
#: Rest tilt: left pad higher than right, so a ball arriving from ramp 1
#: rolls downhill off the right end towards ramp 2 -- derived from the same
#: two pad heights the connecting floor segments use, so the beam's rest
#: position lines up with its neighbours with no visible step.
LEVER_REST_ANGLE = math.atan2(P_LEVER_RIGHT[1] - P_LEVER_LEFT[1], P_LEVER_RIGHT[0] - P_LEVER_LEFT[0])

PULLEY_RADIUS = 0.45


def _floor_y(x: float) -> float:
    t = (x - P_FLOOR_START[0]) / (P_BUCKET_DROP_TOP[0] - P_FLOOR_START[0])
    return P_FLOOR_START[1] + t * (P_BUCKET_DROP_TOP[1] - P_FLOOR_START[1])


#: Sits straddling the single floor segment (half "underground") instead of
#: on its own flat bench, so the ball rolls over its top exactly like
#: cresting a bump -- no gap, no dead-flat stretch for `SPACE_DAMPING` to
#: strand it on.
PULLEY_CENTER = (15.4, _floor_y(15.4))


@dataclass(frozen=True)
class PhysicsSample:
    t: float
    ball_x: float
    ball_y: float
    ball_angle: float
    lever_angle: float
    pulley_angle: float


def _build_space() -> tuple[pymunk.Space, pymunk.Body, pymunk.Body, pymunk.Body]:
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    space.damping = SPACE_DAMPING

    def seg(a: tuple[float, float], b: tuple[float, float], friction: float = 0.95) -> None:
        shape = pymunk.Segment(space.static_body, a, b, 0.03)
        shape.friction = friction
        shape.elasticity = 0.0
        space.add(shape)

    seg(P_FUNNEL_TOP, P_RAMP1_TOP)
    seg(P_RAMP1_TOP, P_RAMP1_BOTTOM)
    seg(P_RAMP1_BOTTOM, P_LEVER_LEFT)
    seg(P_LEVER_RIGHT, P_RAMP2_TOP)
    seg(P_RAMP2_TOP, P_RAMP2_BOTTOM)
    seg(P_FLOOR_START, P_BUCKET_DROP_TOP)
    seg(P_BUCKET_DROP_TOP, P_BUCKET_FLOOR_LEFT)
    seg(P_BUCKET_FLOOR_LEFT, (BUCKET_RIGHT_WALL_X, P_BUCKET_FLOOR_LEFT[1]))
    seg((BUCKET_RIGHT_WALL_X, P_BUCKET_FLOOR_LEFT[1]), (BUCKET_RIGHT_WALL_X, BUCKET_WALL_TOP))

    lever_mass = 1.0
    lever_moment = pymunk.moment_for_segment(lever_mass, (-LEVER_HALF_LEN, 0), (LEVER_HALF_LEN, 0), 0.08)
    lever_body = pymunk.Body(lever_mass, lever_moment)
    lever_body.position = LEVER_PIVOT
    lever_body.angle = LEVER_REST_ANGLE
    lever_shape = pymunk.Segment(lever_body, (-LEVER_HALF_LEN, 0), (LEVER_HALF_LEN, 0), 0.08)
    lever_shape.friction = 0.95
    lever_shape.elasticity = 0.0
    space.add(lever_body, lever_shape)
    pivot_joint = pymunk.PivotJoint(space.static_body, lever_body, LEVER_PIVOT)
    pivot_joint.collide_bodies = False
    # DampedRotarySpring's rest_angle is body_a.angle - body_b.angle; body_a
    # is the static anchor (always angle 0), so the lever's OWN equilibrium
    # angle is the negative of what's passed in here -- flip the sign so it
    # actually settles at LEVER_REST_ANGLE (confirmed empirically against a
    # standalone trace, not derived from the pymunk docs alone).
    spring = pymunk.DampedRotarySpring(
        space.static_body, lever_body, rest_angle=-LEVER_REST_ANGLE, stiffness=35.0, damping=9.0
    )
    space.add(pivot_joint, spring)

    pulley_mass = 0.5
    pulley_moment = pymunk.moment_for_circle(pulley_mass, 0, PULLEY_RADIUS)
    pulley_body = pymunk.Body(pulley_mass, pulley_moment)
    pulley_body.position = PULLEY_CENTER
    pulley_shape = pymunk.Circle(pulley_body, PULLEY_RADIUS)
    pulley_shape.friction = 1.6
    pulley_shape.elasticity = 0.0
    space.add(pulley_body, pulley_shape)
    pulley_pivot = pymunk.PivotJoint(space.static_body, pulley_body, PULLEY_CENTER)
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


def _simulate() -> list[PhysicsSample]:
    space, ball, lever, pulley = _build_space()
    samples: list[PhysicsSample] = []
    rest_streak = 0
    for i in range(MAX_STEPS):
        space.step(DT)
        samples.append(PhysicsSample(i * DT, ball.position.x, ball.position.y, ball.angle, lever.angle, pulley.angle))
        at_rest = abs(ball.velocity.x) < 0.05 and abs(ball.velocity.y) < 0.05
        if ball.position.x > BUCKET_RIGHT_WALL_X - 1.0 and at_rest:
            rest_streak += 1
            if rest_streak > 30:
                break
        else:
            rest_streak = 0
    return samples


#: Computed once, at import time -- see module docstring for why this never
#: needs to depend on (or be recomputed for) the demo's `target` slider.
TRAJECTORY: list[PhysicsSample] = _simulate()

#: The six narrative legs, as (start_x_threshold, end_x_threshold) pairs,
#: ONE PER CLICK -- in the SAME order as `rube_goldberg_chain.py`'s six
#: checkpoint-to-checkpoint motion gaps (funnel drop, ramp 1, lever, ramp
#: 2, pulley, bucket). Each is a half-open slice of `TRAJECTORY` by ball
#: x-position -- the ball never stops moving between them (a real machine
#: doesn't pause between gadgets either), so these are where one gadget's
#: influence on the picture hands off to the next, not places the ball
#: rests. Deliberately six, not five: an earlier version folded the
#: funnel-drop into the ramp-1 leg, which meant the FIRST click of the demo
#: produced no visible motion at all (ball just faded in, standing still)
#: -- confirmed by simulating a click end to end and tracing ball_x across
#: it. Every click must move the ball, including the first one.
LEG_X_BOUNDS: tuple[tuple[float, float], ...] = (
    (P_SPAWN[0], P_RAMP1_TOP[0]),
    (P_RAMP1_TOP[0], P_RAMP1_BOTTOM[0]),
    (P_RAMP1_BOTTOM[0], P_LEVER_RIGHT[0]),
    (P_LEVER_RIGHT[0], P_RAMP2_BOTTOM[0]),
    (P_RAMP2_BOTTOM[0], P_BUCKET_DROP_TOP[0]),
    (P_BUCKET_DROP_TOP[0], BUCKET_RIGHT_WALL_X),
)


def _index_at_or_after_x(x_threshold: float) -> int:
    for i, s in enumerate(TRAJECTORY):
        if s.ball_x >= x_threshold:
            return i
    return len(TRAJECTORY) - 1


def leg_samples(leg_index: int, n: int) -> list[PhysicsSample]:
    """``n`` evenly-spaced physics samples spanning narrative leg ``leg_index``.

    Used to populate a checkpoint-to-checkpoint tween with real simulated
    motion instead of a linear/eased interpolation -- see
    ``rube_goldberg_chain.py``'s ``_physics_tween``.
    """
    x0, x1 = LEG_X_BOUNDS[leg_index]
    start = _index_at_or_after_x(x0)
    end = _index_at_or_after_x(x1)
    if end <= start:
        end = min(start + 1, len(TRAJECTORY) - 1)
    return [TRAJECTORY[round(start + (end - start) * k / max(n - 1, 1))] for k in range(n)]


def leg_boundary_sample(leg_index: int, end: bool) -> PhysicsSample:
    """The physics sample at the start (``end=False``) or end (``end=True``)
    of narrative leg ``leg_index`` -- what a checkpoint's own (non-tween)
    frame shows, so it lines up exactly with the tween that arrives at it."""
    x0, x1 = LEG_X_BOUNDS[leg_index]
    return TRAJECTORY[_index_at_or_after_x(x1 if end else x0)]


def final_sample() -> PhysicsSample:
    """Where the ball actually comes to rest -- the bucket's picture uses
    this instead of a hand-picked position, so it can never disagree with
    where the simulated ball really lands."""
    return TRAJECTORY[-1]
