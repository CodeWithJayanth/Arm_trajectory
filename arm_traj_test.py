import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def parse_list_of_floats(s: str, name: str, expected_len: Optional[int] = None) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} must not be empty")
    if expected_len is not None and len(vals) != expected_len:
        raise ValueError(f"{name} must have {expected_len} values")
    return vals

def parse_q4(s: str) -> Optional[List[float]]:
    if s is None or s.strip() == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("q vector must have exactly 4 comma-separated values: pitch,yaw,roll,elbow")
    try:
        vals = [float(x) for x in parts]
    except ValueError as e:
        raise ValueError("failed to parse q vector; expected 4 floats in radians") from e

    return vals

def parse_optional_per_joint(s: str, dof: int, name: str) -> Optional[np.ndarray]:
    if not s or not s.strip():
        return None
    vals = parse_list_of_floats(s, name)
    arr = np.full(dof, vals[0], dtype=np.float64) if len(vals) == 1 else np.asarray(vals, dtype=np.float64)
    if len(arr) != dof:
        raise ValueError(f"{name} must be one value or {dof} values")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} values must be > 0")
    return arr


def parse_pred_horizons(args) -> List[float]:
    argv = sys.argv[1:]
    hs_set = any(a == "--pred-horizons" or a.startswith("--pred-horizons=") for a in argv)
    h_set = any(a == "--pred-horizon" or a.startswith("--pred-horizon=") for a in argv)
    vals = parse_list_of_floats(args.pred_horizons, "pred-horizons") if hs_set or not h_set else [float(args.pred_horizon)]
    out = sorted(set(v for v in vals if v > 0.0))
    if not out:
        raise ValueError("at least one positive horizon is required")
    return out


def get_start_goal_for_movement(movement: str, lower: np.ndarray, upper: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_base = np.asarray([2.6213, 0.1448, -0.7687, 0.8569], dtype=np.float64)
    
    if movement == "raise_0_to_90":
        q_start, q_goal = q_base.copy(), q_base.copy(); q_goal[0] += 1.5708
    elif movement == "lower_90_to_0":
        q_start, q_goal = q_base.copy(), q_base.copy(); q_start[0] += 1.5708
    elif movement == "sweep_left_to_right":
        q_start, q_goal = q_base.copy(), q_base.copy(); q_start[1] -= 0.7854; q_goal[1] += 0.7854
    elif movement == "sweep_right_to_left":
        q_start, q_goal = q_base.copy(), q_base.copy(); q_start[1] += 0.7854; q_goal[1] -= 0.7854
    else:
        raise ValueError(f"Unknown movement: {movement}")
    return np.clip(q_start, lower, upper), np.clip(q_goal, lower, upper)


def sample_true_goal(q_nominal: np.ndarray, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator,
                     std_scale: float) -> np.ndarray:
    return np.clip(q_nominal + rng.normal(0.0, 1.0, size=q_nominal.shape) * (std_scale * (upper - lower)), lower, upper)


def generate_candidates(
    q_nominal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    K: int,
    cand_std: float,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    dof = int(q_nominal.shape[0])
    goals = np.zeros((K, dof), dtype=np.float64)
    goals[0] = np.clip(q_nominal, lower, upper)

    if K == 1:
        return goals

    scale = cand_std * (upper - lower)
    n = K - 1

    if mode == "gaussian":
        offs = rng.normal(0.0, 1.0, size=(n, dof))
    elif mode == "sobol":
        try:
            from scipy.stats import qmc  # type: ignore
            m = int(math.ceil(math.log2(max(1, n))))
            sob = qmc.Sobol(d=dof, scramble=False)
            u = sob.random_base2(m=m)[:n]
            offs = 2.0 * u - 1.0
        except Exception:
            print("WARNING sobol unavailable; falling back to gaussian candidates")
            offs = rng.normal(0.0, 1.0, size=(n, dof))
    elif mode == "grid":
        n_side = int(math.ceil(n ** (1.0 / dof)))
        levels = np.linspace(-1.0, 1.0, max(2, n_side))
        mesh = np.meshgrid(*([levels] * dof), indexing="ij")
        pts = np.stack([m.reshape(-1) for m in mesh], axis=1)
        offs = pts[:n]
    else:
        raise ValueError(f"Unknown candidate mode: {mode}")

    goals[1:] = np.clip(q_nominal + offs * scale, lower, upper)
    return goals



def min_jerk_single(q0: np.ndarray, dq0: np.ndarray, qf: np.ndarray, T: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    dof = len(q0); T = max(T, 1e-6); n_steps = max(2, int(T / dt) + 1)
    c0 = q0.astype(np.float64); c1 = dq0.astype(np.float64); c2 = np.zeros(dof, dtype=np.float64)
    T2, T3, T4, T5 = T * T, T**3, T**4, T**5
    A = np.array([[T3, T4, T5], [3*T2, 4*T3, 5*T4], [6*T, 12*T2, 20*T3]], dtype=np.float64)
    b = np.array([qf - c0 - c1*T - c2*T2, -c1 - 2.0*c2*T, -2.0*c2], dtype=np.float64)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.zeros((3, dof), dtype=np.float64)
    c3, c4, c5 = x[0], x[1], x[2]
    t = np.linspace(0.0, T, n_steps)[:, None]
    q = c0 + c1*t + c2*(t**2) + c3*(t**3) + c4*(t**4) + c5*(t**5)
    dq = c1 + 2.0*c2*t + 3.0*c3*(t**2) + 4.0*c4*(t**3) + 5.0*c5*(t**4)
    return q, dq


def min_jerk_next(q0: np.ndarray, dq0: np.ndarray, qf: np.ndarray, T_rem: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    q, dq = min_jerk_single(q0, dq0, qf, max(T_rem, dt), dt)
    return q[1].copy(), dq[1].copy()


def apply_joint_delays(q: np.ndarray, dt: float, delays_s: List[float]) -> np.ndarray:
    out = q.copy(); n = q.shape[0]
    for j in range(q.shape[1]):
        d = int(round(delays_s[j] / dt))
        if d <= 0: continue
        if d >= n: out[:, j] = q[0, j]
        else:
            out[:d, j] = q[0, j]
            out[d:, j] = q[:-d, j]
    return out


def apply_kinematic_caps(q_world: np.ndarray, dt: float, lower: np.ndarray, upper: np.ndarray,
                         dq_max: Optional[np.ndarray], ddq_max: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    
    
    if dq_max is None and ddq_max is None:
        return q_world, np.gradient(q_world, dt, axis=0)
    dq = np.gradient(q_world, dt, axis=0)
   
    if dq_max is not None:
        dq = np.clip(dq, -dq_max, dq_max)
   
   
    if ddq_max is not None:
        ddq = np.clip(np.gradient(dq, dt, axis=0), -ddq_max, ddq_max)
        dq_i = np.zeros_like(dq); dq_i[0] = dq[0]
        
        
        for i in range(1, len(dq)):
            dq_i[i] = dq_i[i-1] + ddq[i-1] * dt
        dq = np.clip(dq_i, -dq_max, dq_max) if dq_max is not None else dq_i
   
   
    q_out = np.zeros_like(q_world); q_out[0] = q_world[0]
   
   
    for i in range(1, len(q_world)):
        q_out[i] = q_out[i-1] + dq[i-1] * dt
   
   
    q_out = np.clip(q_out, lower, upper)
    dq_out = np.gradient(q_out, dt, axis=0)
   
   
    if dq_max is not None:
        dq_out = np.clip(dq_out, -dq_max, dq_max)
    return q_out, dq_out



@dataclass
class ObsParams:
    delay_steps: int
    pos_noise: float
    vel_noise: float
class ObsBuffer:
    def __init__(self, delay_steps: int):
        self.delay_steps = max(0, int(delay_steps)); self.q_hist: List[np.ndarray] = []; self.dq_hist: List[np.ndarray] = []
    def push(self, q: np.ndarray, dq: np.ndarray):
        self.q_hist.append(q.copy()); self.dq_hist.append(dq.copy())
    def get_delayed(self) -> Tuple[np.ndarray, np.ndarray]:
        i = max(0, len(self.q_hist) - 1 - self.delay_steps)
        return self.q_hist[i].copy(), self.dq_hist[i].copy()


def add_observation_noise(q: np.ndarray, dq: np.ndarray, rng: np.random.Generator, params: ObsParams):
    return (
        q + rng.normal(0.0, params.pos_noise, size=q.shape),
        dq + rng.normal(0.0, params.vel_noise, size=dq.shape),
    )


def softmax_log(logp: np.ndarray) -> np.ndarray:
    z = logp - np.max(logp); e = np.exp(z); s = np.sum(e)
    return np.full_like(logp, 1.0 / len(logp)) if s <= 0.0 else e / s


def belief_update(logp: np.ndarray, q_obs: np.ndarray, dq_obs: np.ndarray, pred_q_t: np.ndarray, pred_dq_t: np.ndarray,
                  speed: float, sigma_q: float, sigma_dq_base: float, alpha_sdn: float,
                  lik_scale: float, min_speed: float) -> np.ndarray:


    if speed < min_speed:
        return logp


    sq = sigma_q * sigma_q; sdq = (sigma_dq_base + alpha_sdn * speed) ** 2
    e_q = q_obs - pred_q_t; e_dq = dq_obs - pred_dq_t
    ll = -0.5 * np.sum(e_q*e_q / sq, axis=1) - 0.5 * np.sum(e_dq*e_dq / sdq, axis=1)
    ll = np.maximum(ll, -500.0)
    out = logp + lik_scale * ll
    return out - np.max(out)


def rollout_horizon(q0: np.ndarray, dq0: np.ndarray, goal: np.ndarray, T_rem: float, dt: float, h_steps: int):
    dof = int(q0.shape[0]); qh = np.zeros((h_steps, dof)); dqh = np.zeros((h_steps, dof))
    q, dq = q0.copy(), dq0.copy()
    for i in range(h_steps):
        q, dq = min_jerk_next(q, dq, goal, max(dt, T_rem - i*dt), dt)
        qh[i] = q; dqh[i] = dq
    return qh, dqh



def horizon_rmse(q_hat: np.ndarray, dq_hat: np.ndarray, q_true: np.ndarray, dq_true: np.ndarray) -> Tuple[float, float]:
    rmse_q = float(np.sqrt(np.mean(np.sum((q_hat - q_true)**2, axis=1))))
    rmse_dq = float(np.sqrt(np.mean(np.sum((dq_hat - dq_true)**2, axis=1))))
    return rmse_q, rmse_dq


def nearest_candidate_error(goals: np.ndarray, q_true: np.ndarray) -> Tuple[int, float]:
    d = np.linalg.norm(goals - q_true, axis=1); i = int(np.argmin(d))
    return i, float(d[i] * 180.0 / math.pi)

def nearest_candidate_max_joint_error_deg(goals: np.ndarray, q_true: np.ndarray) -> Tuple[int, float]:
    # returns (index, min achievable max-per-joint error in degrees)
    per_joint = np.abs((goals - q_true[None, :]) * (180.0 / math.pi))  
    max_per_goal = np.max(per_joint, axis=1) 
    i = int(np.argmin(max_per_goal))
    return i, float(max_per_goal[i])

def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    pp = np.clip(p, eps, 1.0)
    return float(-np.sum(pp * np.log(pp)))



def replay_traj(env, q_traj: np.ndarray, sleep_s: float) -> bool:
    try:
        for t in range(len(q_traj)):
            env.reset_human_arm(q_traj[t])
            env.bc.stepSimulation(physicsClientId=env.bc._client)
            time.sleep(max(0.0, sleep_s))
    except Exception as e:
        print(f"[Playback] stopped: {e}")
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Minimal hidden-goal inference + multi-horizon forecasting")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--sleep", type=float, default=0.02)
    p.add_argument("--pause", action="store_true")
    p.add_argument("--playback", type=str, default="none", choices=["none", "gt", "pred", "both"])
    p.add_argument("--movement", type=str, default="raise_0_to_90", choices=["raise_0_to_90", "lower_90_to_0", "sweep_left_to_right", "sweep_right_to_left"])
    p.add_argument("--q-start", type=str, default="")
    p.add_argument("--q-goal", type=str, default="")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--joint-delays", type=str, default="0.0,0.03,0.06,0.10")
    p.add_argument("--delay-jitter", type=float, default=0.03)
    p.add_argument("--delay-seed", type=int, default=0)
    p.add_argument("--obs-delay-steps", type=int, default=3)
    p.add_argument("--pos-noise", type=float, default=0.02)
    p.add_argument("--vel-noise", type=float, default=0.08)
    p.add_argument("--cand-std", type=float, default=0.22)
    p.add_argument("--candidate-mode", type=str, default="sobol", choices=["gaussian", "sobol", "grid"])
    p.add_argument("--true-goal-std", type=float, default=0.18)
    p.add_argument("--sigma-q", type=float, default=0.05)
    p.add_argument("--sigma-dq-base", type=float, default=0.15)
    p.add_argument("--alpha-sdn", type=float, default=0.30)
    p.add_argument("--lik-scale", type=float, default=1.0)
    p.add_argument("--min-speed", type=float, default=0.05)
    p.add_argument("--belief-start-step", type=int, default=3)
    p.add_argument("--pred-horizon", type=float, default=0.5)
    p.add_argument("--pred-horizons", type=str, default="0.2,0.5,1.0")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--dq-max", type=str, default="")
    p.add_argument("--ddq-max", type=str, default="")
    args = p.parse_args()
    if args.playback != "none":
        args.gui = True
    return args


def run_single_trial(args, seed: int, env, lower: np.ndarray, upper: np.ndarray) -> dict:
    dt, T = float(args.dt), float(args.duration)
    K = int(args.K)
    if K < 2:
        raise ValueError("--K must be >= 2")


    rng = np.random.default_rng(seed)
    model_delays = parse_list_of_floats(args.joint_delays, "joint-delays", expected_len=4)
    dseed = args.delay_seed if args.delay_seed != 0 else seed
    drng = np.random.RandomState(dseed)
    world_delays = [max(0.0, model_delays[i] + float(drng.uniform(-args.delay_jitter, args.delay_jitter))) for i in range(4)]
    q_start_cli = parse_q4(args.q_start)
    q_goal_cli = parse_q4(args.q_goal)


    if (q_start_cli is None) ^ (q_goal_cli is None):
        raise ValueError("must provide both --q-start and --q-goal together")


    if q_start_cli is not None and q_goal_cli is not None:
        q_start = np.clip(np.asarray(q_start_cli, dtype=np.float64), lower, upper)
        q_nominal = np.clip(np.asarray(q_goal_cli, dtype=np.float64), lower, upper)
        start_goal_source = "cli"
    else:
        q_start, q_nominal = get_start_goal_for_movement(args.movement, lower, upper)
        start_goal_source = "movement"



    q_true = sample_true_goal(q_nominal, lower, upper, rng, std_scale=float(args.true_goal_std))
    q_world, _ = min_jerk_single(q_start, np.zeros_like(q_start), q_true, T, dt)
    q_world = np.clip(q_world, lower, upper)
    q_world = apply_joint_delays(q_world, dt, world_delays)
    q_world = np.clip(q_world, lower, upper)
    dq_max = parse_optional_per_joint(args.dq_max, len(q_start), "dq-max")
    ddq_max = parse_optional_per_joint(args.ddq_max, len(q_start), "ddq-max")
    q_world, dq_world = apply_kinematic_caps(q_world, dt, lower, upper, dq_max, ddq_max)



    N = len(q_world)
    goals = generate_candidates(
        q_nominal=q_nominal,
        lower=lower,
        upper=upper,
        K=K,
        cand_std=float(args.cand_std),
        rng=rng,
        mode=str(args.candidate_mode),
    )



    print("DEBUG cand_count=", len(goals), "K_arg=", args.K, "seed=", seed)
    if q_start_cli is not None and q_goal_cli is not None:
        if not np.allclose(q_start, np.asarray(q_start_cli, dtype=np.float64), atol=1e-9, rtol=0.0):
            print("WARNING q_start mismatch with --q-start (preset override issue)")
        if not np.allclose(q_nominal, np.asarray(q_goal_cli, dtype=np.float64), atol=1e-9, rtol=0.0):
            print("WARNING q_goal_nominal mismatch with --q-goal (preset override issue)")
    cand_mean = np.mean(goals, axis=0)
    cand_std_j = np.std(goals, axis=0)
    cand_min = np.min(goals, axis=0)
    cand_max = np.max(goals, axis=0)


    clipped_mask = np.isclose(goals, lower[None, :], atol=1e-12) | np.isclose(goals, upper[None, :], atol=1e-12)
    clipped_counts = np.sum(clipped_mask, axis=0)
    print(f"q_goal_true (rad): {np.array2string(q_true, precision=4, separator=', ')}")
    print(f"cand_std={float(args.cand_std)} true_goal_std={float(args.true_goal_std)}")
    print(f"K={K}")
    print("candidate_goal_stats:")
    print(f"  mean={np.array2string(cand_mean, precision=4, separator=', ')}")
    print(f"  std={np.array2string(cand_std_j, precision=4, separator=', ')}")
    print(f"  min={np.array2string(cand_min, precision=4, separator=', ')}")
    print(f"  max={np.array2string(cand_max, precision=4, separator=', ')}")
    print(f"  clipped_per_joint={np.array2string(clipped_counts, separator=', ')}")
    mean_thresh = max(0.05, 0.5 * float(args.cand_std))



    for j in range(goals.shape[1]):
        mean_delta = abs(float(cand_mean[j] - q_nominal[j]))
        if mean_delta > mean_thresh:
            print(f"WARNING candidate mean offset joint={j} |mean-goal|={mean_delta:.4f} > {mean_thresh:.4f}")
    min_true_dist = float(np.min(np.linalg.norm(goals - q_true[None, :], axis=1)))



    print(f"candidate_true_goal_min_dist_l2={min_true_dist:.8f}")
    if min_true_dist < 1e-6:
        print("WARNING candidate set contains true goal")
    nn_i, nn_err_deg = nearest_candidate_error(goals, q_true)

    true_idx = nn_i 
    nn_i_max, nn_lb_maxjoint = nearest_candidate_max_joint_error_deg(goals, q_true)


    pred_q = np.zeros((N, K, len(q_start)), dtype=np.float64)
    pred_dq = np.zeros((N, K, len(q_start)), dtype=np.float64)
    for k in range(K):
        qk, _ = min_jerk_single(q_start, np.zeros_like(q_start), goals[k], T, dt)
        qk = np.clip(apply_joint_delays(np.clip(qk, lower, upper), dt, model_delays), lower, upper)
        pred_q[:, k, :] = qk
        pred_dq[:, k, :] = np.gradient(qk, dt, axis=0)
    
    
    
    obs = ObsParams(delay_steps=int(args.obs_delay_steps), pos_noise=float(args.pos_noise), vel_noise=float(args.vel_noise))
    obs_buf = ObsBuffer(obs.delay_steps)
    obs_rng = np.random.default_rng(seed + 999)
    horizon_secs = parse_pred_horizons(args)
    horizon_steps = [int(max(1, round(h / dt))) for h in horizon_secs]
    store_q: Dict[int, Dict[int, np.ndarray]] = {}
    store_dq: Dict[int, Dict[int, np.ndarray]] = {}
    scored_q: Dict[int, List[float]] = {hs: [] for hs in horizon_steps}
    scored_dq: Dict[int, List[float]] = {hs: [] for hs in horizon_steps}
    scored_n: Dict[int, int] = {hs: 0 for hs in horizon_steps}
    logp = np.full(K, -math.log(K), dtype=np.float64)
    commit_time = None
    print("\n" + "=" * 72)
    print(f"Trial seed={seed} movement={args.movement} K={K} dt={dt:.3f} T={T:.2f}")
    print(f"start_goal_source={start_goal_source} candidate_mode={args.candidate_mode}")
    print(f"q_start (rad): {np.array2string(q_start, precision=4, separator=', ')}")
    print(f"q_goal_nominal (rad): {np.array2string(q_nominal, precision=4, separator=', ')}")
    print(f"NN lower-bound={nn_err_deg:.2f}° idx={nn_i}")

    

    for t_idx in range(N):
        t_sec = t_idx * dt
        env.reset_human_arm(q_world[t_idx])
        if args.gui:
            env.bc.stepSimulation(physicsClientId=env.bc._client)
            time.sleep(max(0.0, float(args.sleep)))
        obs_buf.push(q_world[t_idx], dq_world[t_idx])
        qd, dqd = obs_buf.get_delayed()
        q_obs, dq_obs = add_observation_noise(qd, dqd, obs_rng, obs)
        if t_idx >= int(args.belief_start_step):
            pi = max(0, t_idx - obs.delay_steps)
            logp = belief_update(logp, q_obs, dq_obs, pred_q[pi], pred_dq[pi], float(np.linalg.norm(dq_obs)),
                                float(args.sigma_q), float(args.sigma_dq_base), float(args.alpha_sdn),
                                float(args.lik_scale), float(args.min_speed))
        p_now = softmax_log(logp)
        if commit_time is None and float(np.max(p_now)) > 0.80:
            commit_time = t_sec
        best_i = int(np.argmax(p_now))
        goal_est = q_true.copy() if args.oracle else goals[best_i].copy()
        obs_idx = max(0, t_idx - obs.delay_steps)
        T_rem = max(dt, T - obs_idx * dt)
        store_q[t_idx] = {}; store_dq[t_idx] = {}
        for hs in horizon_steps:
            qh, dqh = rollout_horizon(q_obs.copy(), dq_obs.copy(), goal_est, T_rem, dt, hs)
            store_q[t_idx][hs] = qh
            store_dq[t_idx][hs] = dqh


    for t_idx in sorted(store_q.keys()):
        start = t_idx + 1
        for hs in horizon_steps:
            end = start + hs
            if end <= N:
                rq, rdq = horizon_rmse(store_q[t_idx][hs], store_dq[t_idx][hs], q_world[start:end], dq_world[start:end])
                scored_q[hs].append(rq); scored_dq[hs].append(rdq); scored_n[hs] += 1
    p_fin = softmax_log(logp)
    best_fin = int(np.argmax(p_fin))

    order = np.argsort(-logp)
    gap = float(logp[order[0]] - logp[order[1]])
    print(f"logp_gap(best-second)={gap:.2f}")

    # Ranking / top-k diagnostics
    order = np.argsort(-p_fin)  # descending
    top1 = int(order[0])
    top5 = set(int(x) for x in order[:5])

    top1_hits_true = (top1 == true_idx)
    top5_hits_true = (true_idx in top5)

    # Baselines
    nom_idx = 0
    rand_idx = int(rng.integers(0, K))

    def goal_err_stats_deg(idx: int) -> Tuple[float, float]:
        pj = np.abs((goals[idx] - q_true) * (180.0 / math.pi))
        return float(np.mean(pj)), float(np.max(pj))

    nom_mean, nom_max = goal_err_stats_deg(nom_idx)
    rand_mean, rand_max = goal_err_stats_deg(rand_idx)

    # Belief calibration
    pmax = float(np.max(p_fin))
    H = entropy(p_fin)


    q_pred_goal = goals[best_fin].copy()
    per_joint_deg = np.abs((q_pred_goal - q_true) * (180.0 / math.pi))
    mean_err, max_err = float(np.mean(per_joint_deg)), float(np.max(per_joint_deg))
    passed = max_err <= 10.0


    q_rmse_by_h = {f"{h:.2f}": (float(np.mean(scored_q[hs])) if scored_q[hs] else None) for h, hs in zip(horizon_secs, horizon_steps)}
    dq_rmse_by_h = {f"{h:.2f}": (float(np.mean(scored_dq[hs])) if scored_dq[hs] else None) for h, hs in zip(horizon_secs, horizon_steps)}
    n_by_h = {f"{h:.2f}": scored_n[hs] for h, hs in zip(horizon_secs, horizon_steps)}
    
    
    
    
    print("FINAL")
    print(f"goal mean={mean_err:.2f}° max={max_err:.2f}° pass={passed}")

    print(f"true_idx(L2 NN)={true_idx}  pred_idx={best_fin}  top1_hit={top1_hits_true} top5_hit={top5_hits_true}")
    print(f"LB max-joint (achievable)={nn_lb_maxjoint:.2f}°  (idx={nn_i_max})")
    print(f"baseline nominal(idx=0): mean={nom_mean:.2f}° max={nom_max:.2f}°")
    print(f"baseline random(idx={rand_idx}): mean={rand_mean:.2f}° max={rand_max:.2f}°")
    print(f"belief: pmax={pmax:.3f} entropy={H:.3f}")

    print(f"commit_time={('%.2fs' % commit_time) if commit_time is not None else 'never'}")


    for h, hs in zip(horizon_secs, horizon_steps):
        key = f"{h:.2f}"; rq, rdq = q_rmse_by_h[key], dq_rmse_by_h[key]
        if rq is None: print(f"H={h:.2f}s not_scored")
        else:
            q_deg = (rq / math.sqrt(len(q_start))) * (180.0 / math.pi)
            dq_deg = (rdq / math.sqrt(len(q_start))) * (180.0 / math.pi)
            print(f"H={h:.2f}s scored={n_by_h[key]} RMSE(q)={rq:.5f}rad (~{q_deg:.2f}deg/joint) RMSE(dq)={rdq:.5f}rad/s (~{dq_deg:.2f}deg/s/joint)")


    if args.gui and args.playback in {"gt", "pred", "both"}:
        s = max(0.0, float(args.sleep))
        if args.playback in {"gt", "both"}:
            print("[Playback] GT")
            replay_traj(env, q_world, s)
        if args.playback in {"pred", "both"}:
            rg = q_true if args.oracle else q_pred_goal
            q_pred_nom, _ = min_jerk_single(q_start, np.zeros_like(q_start), rg, T, dt)
            q_pred_nom = np.clip(apply_joint_delays(np.clip(q_pred_nom, lower, upper), dt, model_delays), lower, upper)
            print("[Playback] Predicted nominal")
            replay_traj(env, q_pred_nom, s)
    return {
        "mean_err": mean_err,
        "max_err": max_err,
        "passed": passed,
        "commit_time": commit_time,
        "forecast_rmse_q_mean_by_h": q_rmse_by_h,
        "forecast_rmse_dq_mean_by_h": dq_rmse_by_h,
        "forecast_n_scored_by_h": n_by_h,
        "nn_lb_maxjoint_deg": nn_lb_maxjoint,
    }



def main():
    args = parse_args()
    from manip4care.envs.manipulation_env import ManipulationEnv
    from manip4care.envs.wiping_env import WipingEnv
    env = None; wipe = None


    try:
        env = ManipulationEnv(gui=args.gui, wiping=False)
        wipe = WipingEnv()
        env.reset(); wipe.reset()
        lower = np.asarray(wipe.human_arm_lower_limits, dtype=np.float64)
        upper = np.asarray(wipe.human_arm_upper_limits, dtype=np.float64)
        n_trials = max(1, int(args.trials)); results = []


        for tr in range(n_trials):
            results.append(run_single_trial(args, int(args.seed) + tr, env, lower, upper))
        pass_rate = 100.0 * sum(1 for r in results if r["passed"]) / float(n_trials)


        print("\n" + "=" * 72)
        print(f"SUMMARY trials={n_trials}")
        print(f"goal mean={np.mean([r['mean_err'] for r in results]):.2f}°")
        print(f"goal max={np.mean([r['max_err'] for r in results]):.2f}°")
        print(f"pass_rate={pass_rate:.1f}%")
        print("=" * 72)
        if args.gui and args.pause:
            print("Paused. Close GUI or Ctrl+C.")
            while True:
                time.sleep(1.0)


    except KeyboardInterrupt:
        pass
    finally:
        try:
            if env is not None:
                env.bc.disconnect()
        except Exception:
            pass
        try:
            if wipe is not None:
                wipe.bc.disconnect()
        except Exception:
            pass
if __name__ == "__main__":
    main()
