# -*- coding: utf-8 -*-
"""
Eigen: Entry-Descent-Landing (EDL) Core

A Mars lander governed by a single geometric invariant ds².
No mode switching, no trajectory planner, no state machine.

State vector: [altitude, downrange, crossrange, v_vertical, v_horizontal_x, v_horizontal_y]
Control: [thrust_magnitude, thrust_pitch, thrust_yaw]

ds² = ‖state - target_state‖²_W + G_terrain·max(0, r - d_terrain)²
      + G_vel·max(0, |v| - v_max)² + G_fuel·(1/(fuel+ε))
      + λ‖u‖²

The system acts on ∇ds². The observer reconstructs ds².
Entry, descent, and landing are not three modes — they are one flow
reaching eigenstate at the surface.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------- physical constants ----------
MARS_G = 3.72076         # m/s², Mars surface gravity
RHO0_MARS = 0.020        # kg/m³, Mars surface atmospheric density
SCALE_HEIGHT = 11100.0   # m, Mars atmosphere scale height


def mars_atmosphere(alt: float) -> float:
    """Atmospheric density at altitude (exponential model)."""
    if alt < 0:
        alt = 0.0
    return RHO0_MARS * np.exp(-alt / SCALE_HEIGHT)


def drag_accel(alt: float, v: np.ndarray, Cd: float = 1.5, A: float = 15.0,
               mass: float = 900.0) -> np.ndarray:
    """
    Aerodynamic drag acceleration: a_drag = -0.5 * rho * Cd * A / m * |v| * v
    Points opposite to velocity. All units SI.
    """
    rho = mars_atmosphere(alt)
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        return np.zeros(3)
    return -0.5 * rho * Cd * A / mass * speed * v


def dynamics(state: np.ndarray, thrust_vec: np.ndarray, fuel: float,
             Cd: float = 1.5, A: float = 15.0, mass: float = 900.0) -> np.ndarray:
    """
    Equations of motion for the lander. All in acceleration units.

    state: [alt, downrange, crossrange, vz, vx, vy]
      alt = altitude above surface (positive up)
      vz  = vertical velocity (positive up)
      vx, vy = horizontal velocities

    thrust_vec: [az, ax, ay] thrust acceleration (m/s²). Positive az = upward.
    fuel: remaining fuel mass (kg). If 0, thrust is zero.

    Returns: d(state)/dt
    """
    alt, dr, cr, vz, vx, vy = state
    vel = np.array([vz, vx, vy])

    # Gravity (downward = negative vz direction)
    grav = np.array([-MARS_G, 0.0, 0.0])

    # Drag acceleration
    ad = drag_accel(max(alt, 0.0), vel, Cd, A, mass)

    # Thrust (zero if no fuel)
    if fuel <= 0:
        thrust_vec = np.zeros(3)

    accel = grav + ad + thrust_vec

    # d/dt [alt, dr, cr, vz, vx, vy]
    return np.array([vz, vx, vy, accel[0], accel[1], accel[2]])


# ---------- ds² computation ----------

def compute_ds2_edl(
    state: np.ndarray,
    control: np.ndarray,
    fuel: float,
    target_state: np.ndarray,
    terrain_hazards: Optional[List[Tuple[np.ndarray, float]]] = None,
    W_pos: float = 1.0,
    W_vel: float = 0.5,
    Go_terrain: float = 50.0,
    Go_vel: float = 10.0,
    v_max: float = 2.0,
    lam: float = 0.01,
    fuel_weight: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute ds² for EDL.

    state: [alt, dr, cr, vz, vx, vy]
    control: [Fz, Fx, Fy] thrust command
    target_state: desired final state [alt=0, dr_t, cr_t, vz≈0, vx≈0, vy≈0]
    terrain_hazards: list of (position_2d, radius) — boulders/slopes to avoid
    v_max: maximum safe landing velocity magnitude
    """
    if terrain_hazards is None:
        terrain_hazards = []

    pos = state[:3]  # [alt, dr, cr]
    vel = state[3:]  # [vz, vx, vy]
    target_pos = target_state[:3]
    target_vel = target_state[3:]

    # Position term (weighted)
    pos_err = pos - target_pos
    target_term = float(W_pos * np.sum(pos_err ** 2))

    # Velocity term
    vel_err = vel - target_vel
    vel_match_term = float(W_vel * np.sum(vel_err ** 2))

    # Terrain hazard avoidance (horizontal plane only)
    terrain_term = 0.0
    d_terrain_min = float('inf')
    horizontal_pos = state[1:3]  # [dr, cr]
    for center_2d, radius in terrain_hazards:
        center_2d = np.asarray(center_2d, dtype=float)
        d = float(np.linalg.norm(horizontal_pos - center_2d))
        d_terrain_min = min(d_terrain_min, d)
        penetration = max(0.0, radius - d)
        terrain_term += Go_terrain * penetration ** 2

    # Velocity limit term
    speed = float(np.linalg.norm(vel))
    vel_excess = max(0.0, speed - v_max)
    vel_limit_term = float(Go_vel * vel_excess ** 2)

    # Fuel term: as fuel depletes, cost of thrusting increases
    fuel_term = 0.0
    if fuel_weight > 0:
        fuel_term = float(fuel_weight / (fuel + 1.0))

    # Control regularization
    reg_term = float(lam * np.sum(control ** 2))

    ds2 = target_term + vel_match_term + terrain_term + vel_limit_term + fuel_term + reg_term

    return ds2, {
        "target_term": target_term,
        "vel_match_term": vel_match_term,
        "terrain_term": terrain_term,
        "vel_limit_term": vel_limit_term,
        "fuel_term": fuel_term,
        "reg_term": reg_term,
        "d_terrain_min": d_terrain_min,
        "speed": speed,
    }


def compute_gradient_edl(
    state: np.ndarray,
    control: np.ndarray,
    fuel: float,
    target_state: np.ndarray,
    terrain_hazards: Optional[List[Tuple[np.ndarray, float]]] = None,
    W_pos: float = 1.0,
    W_vel: float = 0.5,
    Go_terrain: float = 50.0,
    Go_vel: float = 10.0,
    v_max: float = 2.0,
    lam: float = 0.01,
    fuel_weight: float = 0.0,
    dt: float = 0.1,
    eps: float = 1e-4,
    Cd: float = 1.5,
    A: float = 15.0,
    horizon: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    Compute ∇ds² with respect to control, via multi-step lookahead.

    Rolls the dynamics forward `horizon` steps under constant control
    to evaluate ds² of the resulting state. This gives the gradient
    enough reach to see how control choices propagate through position.

    The system acts on this gradient. It never sees ds² itself.

    Returns: (grad, grad_norm)
    """
    if terrain_hazards is None:
        terrain_hazards = []

    def ds2_after_rollout(ctrl):
        s = state.copy()
        f = fuel
        g0 = 9.80665
        for _ in range(horizon):
            # Clamp if out of fuel
            c = ctrl if f > 0 else np.zeros(3)
            deriv = dynamics(s, c, f, Cd, A)
            s = s + dt * deriv
            # fuel burn
            tmag = np.linalg.norm(ctrl)
            if f > 0 and tmag > 0:
                f = max(0.0, f - tmag * dt / (220.0 * g0))
            # ground clamp
            if s[0] < 0:
                s[0] = 0.0
                s[3] = min(s[3], 0.0)
                break
        val, _ = compute_ds2_edl(
            s, ctrl, f, target_state, terrain_hazards,
            W_pos, W_vel, Go_terrain, Go_vel, v_max, lam, fuel_weight
        )
        return val

    base = ds2_after_rollout(control)

    grad = np.zeros(3)
    for i in range(3):
        ctrl_plus = control.copy()
        ctrl_plus[i] += eps
        grad[i] = (ds2_after_rollout(ctrl_plus) - base) / eps

    grad_norm = float(np.linalg.norm(grad))
    return grad, grad_norm


def run_edl_simulation(
    state_init: np.ndarray = None,
    target_state: np.ndarray = None,
    terrain_hazards: Optional[List[Tuple[np.ndarray, float]]] = None,
    n_ticks: int = 1000,
    dt: float = 0.1,
    eta: float = 0.5,
    fuel_init: float = 200.0,
    thrust_max: float = 15.0,
    Isp: float = 220.0,
    W_pos: float = 1.0,
    W_vel: float = 0.5,
    Go_terrain: float = 50.0,
    Go_vel: float = 10.0,
    v_max: float = 2.0,
    lam: float = 0.01,
    fuel_weight: float = 0.0,
    Cd: float = 1.5,
    A: float = 15.0,
    sensor_mask: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Run Eigen-EDL: gradient descent on ds² for Mars landing.

    Euler integration. The system acts on ∇ds² each tick.
    No mode switching. One flow.

    Parameters
    ----------
    state_init : initial [alt, dr, cr, vz, vx, vy]
    target_state : desired final state
    terrain_hazards : list of (center_2d, radius) obstacles on surface
    n_ticks : number of simulation steps
    dt : time step (seconds)
    eta : gradient descent step size for control update
    fuel_init : initial fuel mass (kg)
    thrust_max : max thrust magnitude (N per unit mass, i.e. acceleration)
    Isp : specific impulse for fuel consumption
    sensor_mask : 6-element binary array; 0 = that state channel is dropped (sensor dropout)

    Returns
    -------
    trace : list of dicts with full diagnostic info per tick
    """
    if state_init is None:
        state_init = np.array([2000.0, 0.0, 0.0, -80.0, 10.0, 0.0])
    if target_state is None:
        target_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if terrain_hazards is None:
        terrain_hazards = []
    if sensor_mask is None:
        sensor_mask = np.ones(6)

    state = np.array(state_init, dtype=float)
    fuel = float(fuel_init)
    control = np.zeros(3)  # start with no thrust
    trace = []

    for t in range(n_ticks):
        # Apply sensor mask (dropout simulation)
        sensed_state = state * sensor_mask

        # Compute ∇ds² on sensed state
        grad, gnorm = compute_gradient_edl(
            sensed_state, control, fuel, target_state, terrain_hazards,
            W_pos, W_vel, Go_terrain, Go_vel, v_max, lam, fuel_weight,
            dt=dt, Cd=Cd, A=A,
        )

        # Update control via gradient descent
        control_new = control - eta * grad

        # Clamp thrust magnitude
        thrust_mag = np.linalg.norm(control_new)
        if thrust_mag > thrust_max:
            control_new = control_new * (thrust_max / thrust_mag)
            thrust_mag = thrust_max

        # Compute ds² for diagnostics (observer quantity)
        ds2, comp = compute_ds2_edl(
            sensed_state, control_new, fuel, target_state, terrain_hazards,
            W_pos, W_vel, Go_terrain, Go_vel, v_max, lam, fuel_weight
        )

        # Record
        trace.append({
            "t": t,
            "time": t * dt,
            "state": state.copy(),
            "sensed_state": sensed_state.copy(),
            "control": control_new.copy(),
            "thrust_mag": float(np.linalg.norm(control_new)),
            "ds2": ds2,
            "grad_norm": gnorm,
            "fuel": fuel,
            "altitude": state[0],
            "speed": float(np.linalg.norm(state[3:])),
            "vertical_speed": state[3],
            "components": comp,
        })

        # Check termination: landed
        if state[0] <= 0.0:
            break

        # Physics step (Euler)
        deriv = dynamics(state, control_new, fuel, Cd, A)
        state = state + dt * deriv

        # Fuel consumption: dm/dt = |F| / (Isp * g0)
        g0 = 9.80665
        if fuel > 0 and thrust_mag > 0:
            fuel_consumed = thrust_mag * dt / (Isp * g0)
            fuel = max(0.0, fuel - fuel_consumed)

        # Ground clamp
        if state[0] < 0.0:
            state[0] = 0.0
            state[3] = min(state[3], 0.0)  # no bouncing

        control = control_new

    return trace
