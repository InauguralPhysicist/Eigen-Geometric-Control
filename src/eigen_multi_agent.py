"""
Multi-Agent Paradox-Aware Coordination

Demonstrates how multiple agents can coordinate through a shared lightlike
observer without explicit communication protocols. Coordination emerges from
geometric paradox resolution.

Key insight:
- Each agent: own timelike/spacelike state
- Shared lightlike observer at collective ds² = 0
- Conflicting goals create paradox
- Lightlike observer mediates → emergent coordination

This proves the calculus scales: N agents, 1 observer, emergent behavior.
"""

from typing import List, Tuple

import numpy as np

from .eigen_similarity import detect_oscillation, lorentz_similarity


def collective_lightlike_observer(
    agent_states: List[np.ndarray],
    agent_histories: List[List[np.ndarray]],
    threshold: float = 0.92,
) -> Tuple[bool, List[bool], float]:
    """
    Shared lightlike observer for multiple agents.

    The observer sits at the collective ds² = 0 boundary, measuring
    whether the SYSTEM (all agents together) is in equilibrium.

    Args:
        agent_states: Current state of each agent [agent_0, agent_1, ...]
        agent_histories: Recent trajectory for each agent
        threshold: Oscillation detection threshold

    Returns:
        - collective_oscillating: bool - Is the system oscillating?
        - agent_oscillating: List[bool] - Which agents are oscillating?
        - collective_similarity: float - System-wide similarity measure
    """
    if len(agent_states) == 0:
        return False, [], 0.0

    # Check each agent individually
    agent_oscillating = []
    agent_similarities = []

    for i, (state, history) in enumerate(zip(agent_states, agent_histories)):
        if len(history) >= 2:
            osc, sim = detect_oscillation(history, window=3, threshold=threshold)
            agent_oscillating.append(osc)
            agent_similarities.append(sim)
        else:
            agent_oscillating.append(False)
            agent_similarities.append(0.0)

    # Collective measurement: Are agents interfering with each other?
    # This is the key - we measure PAIRWISE similarity between agents
    # High pairwise similarity = agents are in conflict (oscillating together)
    pairwise_similarities = []

    for i in range(len(agent_states)):
        for j in range(i + 1, len(agent_states)):
            # Are agents i and j in similar states? (potential conflict)
            if len(agent_histories[i]) >= 1 and len(agent_histories[j]) >= 1:
                # Compare recent states
                recent_i = agent_histories[i][-1]
                recent_j = agent_histories[j][-1]

                # Cosine similarity
                dot = np.dot(recent_i, recent_j)
                norm_i = np.linalg.norm(recent_i)
                norm_j = np.linalg.norm(recent_j)

                if norm_i > 1e-9 and norm_j > 1e-9:
                    similarity = dot / (norm_i * norm_j)
                    pairwise_similarities.append(abs(similarity))

    # Collective oscillation: Either individual agents oscillating OR
    # high pairwise similarity (agents interfering)
    individual_osc = any(agent_oscillating)

    if len(pairwise_similarities) > 0:
        collective_similarity = max(pairwise_similarities)
        pairwise_conflict = collective_similarity > threshold
    else:
        collective_similarity = 0.0
        pairwise_conflict = False

    collective_oscillating = individual_osc or pairwise_conflict

    return collective_oscillating, agent_oscillating, float(collective_similarity)


def paradox_mediated_damping(
    agent_idx: int,
    collective_oscillating: bool,
    agent_oscillating: List[bool],
    collective_similarity: float,
    max_damping: float = 0.5,
) -> float:
    """
    Compute damping for a specific agent based on collective state.

    The paradox: Agent wants to reach its goal (individual truth)
                 System wants collective equilibrium (collective truth)

    Lightlike observer mediates by applying MORE damping when there's
    collective conflict, even if the individual agent isn't oscillating.

    Args:
        agent_idx: Which agent to compute damping for
        collective_oscillating: Is the whole system oscillating?
        agent_oscillating: List of per-agent oscillation flags
        collective_similarity: Measure of inter-agent conflict
        max_damping: Maximum damping to apply

    Returns:
        damping: float in [0, max_damping]
    """
    # Individual contribution
    if agent_oscillating[agent_idx]:
        individual_damping = max_damping * 0.5
    else:
        individual_damping = 0.0

    # Collective contribution (this is the key for coordination)
    if collective_oscillating:
        # There's system-wide conflict
        # Apply damping proportional to collective similarity
        # High similarity = high conflict = more damping needed
        collective_damping = max_damping * min(1.0, collective_similarity)
    else:
        collective_damping = 0.0

    # Total damping is max of individual and collective
    # This ensures that even if individual agent is fine,
    # it gets damped if causing system-wide issues
    total_damping = max(individual_damping, collective_damping)

    return float(np.clip(total_damping, 0.0, max_damping))


def multi_agent_step(
    agent_states: List[np.ndarray],
    agent_targets: List[np.ndarray],
    agent_histories: List[List[np.ndarray]],
    eta: float = 0.1,
    max_damping: float = 0.5,
    observer_threshold: float = 0.92,
) -> Tuple[List[np.ndarray], dict]:
    """
    Single step of multi-agent coordination via shared lightlike observer.

    This is the complete paradox-mediated coordination:
    1. Each agent pursues its own goal (individual truth)
    2. Shared observer detects collective conflicts (paradox)
    3. Damping applied to maintain system equilibrium (mediation)
    4. Coordination emerges without explicit communication

    Args:
        agent_states: Current state of each agent
        agent_targets: Target for each agent
        agent_histories: Trajectory history for each agent
        eta: Learning rate for gradient descent
        max_damping: Maximum damping factor
        observer_threshold: Threshold for oscillation detection

    Returns:
        - new_states: Updated agent states after this step
        - collective_info: Diagnostic information about coordination
    """
    n_agents = len(agent_states)

    # 1. SHARED LIGHTLIKE OBSERVER
    # The observer sits at ds² = 0 and measures collective state
    collective_osc, agent_osc, collective_sim = collective_lightlike_observer(
        agent_states, agent_histories, threshold=observer_threshold
    )

    # 2. COMPUTE INDIVIDUAL GRADIENTS
    # Each agent wants to reach its target (individual goals)
    gradients = []
    for state, target in zip(agent_states, agent_targets):
        direction = target - state
        gradients.append(direction)

    # 3. PARADOX-MEDIATED DAMPING
    # Lightlike observer imposes damping to resolve conflicts
    dampings = []
    for i in range(n_agents):
        damping = paradox_mediated_damping(
            i, collective_osc, agent_osc, collective_sim, max_damping
        )
        dampings.append(damping)

    # 4. UPDATE AGENT STATES
    # Apply gradient descent with paradox-mediated damping
    new_states = []
    for i, (state, gradient, damping) in enumerate(zip(agent_states, gradients, dampings)):
        # Gradient step
        velocity = eta * gradient

        # Apply damping (lightlike observer imposing equilibrium)
        velocity = velocity * (1.0 - damping)

        new_state = state + velocity
        new_states.append(new_state)

    # 5. DIAGNOSTIC INFORMATION
    collective_info = {
        "collective_oscillating": collective_osc,
        "agent_oscillating": agent_osc,
        "collective_similarity": collective_sim,
        "dampings": dampings,
        "n_agents_damped": sum(1 for d in dampings if d > 0.01),
        "coordination_active": collective_osc,
    }

    return new_states, collective_info


def check_convergence(
    agent_states: List[np.ndarray], agent_targets: List[np.ndarray], tolerance: float
) -> Tuple[bool, List[float]]:
    """
    Check if all agents have converged to their targets.

    Args:
        agent_states: Current agent states
        agent_targets: Target states
        tolerance: Distance threshold for convergence

    Returns:
        - all_converged: bool - Have all agents reached their targets?
        - distances: List[float] - Distance of each agent to its target
    """
    distances = [
        float(np.linalg.norm(state - target)) for state, target in zip(agent_states, agent_targets)
    ]

    all_converged = all(d < tolerance for d in distances)

    return all_converged, distances


def simulate_multi_agent(
    initial_states: List[np.ndarray],
    targets: List[np.ndarray],
    max_iterations: int = 200,
    eta: float = 0.1,
    max_damping: float = 0.5,
    tolerance: float = 0.1,
) -> Tuple[List[List[np.ndarray]], List[dict], bool]:
    """
    Run full multi-agent simulation with paradox-mediated coordination.

    Args:
        initial_states: Starting state for each agent
        targets: Target state for each agent
        max_iterations: Maximum simulation steps
        eta: Learning rate
        max_damping: Maximum damping factor
        tolerance: Convergence tolerance

    Returns:
        - trajectories: List of trajectories, one per agent
        - info_history: Diagnostic info at each step
        - converged: Did all agents reach their targets?
    """
    n_agents = len(initial_states)

    # Initialize
    current_states = [s.copy() for s in initial_states]
    trajectories = [[s.copy()] for s in initial_states]
    info_history = []

    converged = False

    for iteration in range(max_iterations):
        # Get histories for each agent (last 10 steps)
        agent_histories = [traj[-10:] for traj in trajectories]

        # Multi-agent step with shared lightlike observer
        new_states, info = multi_agent_step(
            current_states,
            targets,
            agent_histories,
            eta=eta,
            max_damping=max_damping,
        )

        # Update
        current_states = new_states
        for i, state in enumerate(new_states):
            trajectories[i].append(state.copy())

        info_history.append(info)

        # Check convergence
        all_converged, distances = check_convergence(current_states, targets, tolerance)

        if all_converged:
            converged = True
            break

    return trajectories, info_history, converged
