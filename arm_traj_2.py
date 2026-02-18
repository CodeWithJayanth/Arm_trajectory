"""
  OFFLINE: Sample styles then  Optimize trajectories (L-BFGS-B) then  Fit ProMP
  ONLINE:  Observe joints then Infer duration then  Condition ProMP then Predict ahead
"""

import numpy as np
import math
from scipy.optimize import minimize

CONFIG = {
    # Task
    "q_start": [0.0, 0.0, 0.3, 0.0],      # start joint angles (4 joints)
    "q_goal":  [0.8, 0.6, 1.2, 0.3],       # goal joint angles
    "dt": 0.02,                              # timestep (50 Hz)

    # Ground truth for testing
    "test_duration": 5.0,                    # seconds of arm movement

    # Library generation
    "num_demos": 100,                        # how many demo trajectories to generate
    "demo_seed": 42,                         # random seed for reproducibility
    "solve_steps": 40,                       # timesteps per optimization

    # Style parameter ranges (Latin Hypercube sampling bounds)
    "style_ranges": {
        "effort_weight":   (0.3, 3.0),
        "speed_weight":    (0.3, 3.0),
        "limit_weight":    (0.3, 3.0),
        "shoulder_ratio":  (0.0, 1.0),
        "goal_weight":     (0.5, 3.0),
        "comfort_weight":  (0.3, 3.0),
        "duration":        (0.8, 6.0),
    },

    # ProMP
    "num_basis": 25,                         # Gaussian basis functions
    "basis_width": 0.5,                      # overlap factor
    "obs_noise": 0.004,                      # observation noise std (rad)
    "phase_points": 200,                     # resolution of phase grid

    # Online temporal inference
    "T_min": 0.8,                            # fastest duration hypothesis
    "T_max": 12.0,                           # slowest duration hypothesis
    "num_T_hypotheses": 30,                  # number of parallel hypotheses
    "condition_window": 5,                   # frames to condition on

    # Evaluation
    "lookahead_horizons": [0.02, 0.1, 0.2, 0.5, 1.0],   # seconds
}

# Joint limits (shoulder1, shoulder2, elbow1, elbow2)
JOINT_LIMITS = np.array([
    [-3.14156,  3.14156],   # sh_yaw (shoulder flexion)
    [-1.00031,  1.46972],   # sh_pitch (shoulder rotation)
    [-3.14105,  3.14105],   # sh_roll (shoulder abduction)
    [ 0.00000,  2.85735],   # elbow
])

# Comfort rest positions (3 postures)


COMFORT_POSES = np.array([
    [ 0.0,  0.2,  0.5,  0.0],
    [-0.2,  0.3,  0.8, -0.1],
    [ 0.1,  0.1,  0.3,  0.1],
])

SHOULDER = np.array([0, 1])
ELBOW = np.array([2, 3])



#  HELPER FUNCTIONS


def min_jerk(s):
    """Min-jerk interpolation: s in [0,1] -> h in [0,1]."""
    return 10*s**3 - 15*s**4 + 6*s**5


def min_jerk_trajectory(q0, qf, T, N):
    """Smooth trajectory from q0 to qf in time T with N steps."""
    tau = np.linspace(0, 1, N + 1)
    s = min_jerk(tau)
    ds = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    dds = (60*tau - 180*tau**2 + 120*tau**3) / T**2

    q = q0 + s[:, None] * (qf - q0)
    qd = ds[:, None] * (qf - q0)
    qdd = dds[:, None] * (qf - q0)
    return q, qd, qdd


def finite_diff_vel(q, dt):
    """Velocity from positions using central differences."""
    v = np.zeros_like(q)
    v[0] = (q[1] - q[0]) / dt
    v[-1] = (q[-1] - q[-2]) / dt
    v[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
    return v


def lhs_samples(num_samples, low, high, rng):
    """Latin Hypercube: num_samples stratified samples in [low, high]."""
    edges = np.linspace(0, 1, num_samples + 1)
    noise = rng.random(num_samples)
    samples = edges[:num_samples] + noise * (edges[1:] - edges[:num_samples])
    rng.shuffle(samples)
    return low + (high - low) * samples


def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()





#  REFERENCE COSTS (for normalizing the style cost function)


def compute_ref_costs(q0, qf):
   
    N = CONFIG["solve_steps"]
    q, qd, qdd = min_jerk_trajectory(q0, qf, 2.0, N)
    eps = 0.05
    qc = COMFORT_POSES[0]

    sums = {"goal": 0, "speed": 0, "effort": 0, "comfort": 0,
            "shoulder": 0, "elbow": 0, "jlimit": 0}

    for t in range(N):
        qt, qdt, u = q[t+1], qd[t+1], qdd[t]
        sums["goal"]     += np.sum((qt - qf)**2)
        sums["speed"]    += np.sum(qdt**2)
        sums["effort"]   += np.sum(u**2)
        sums["comfort"]  += np.sum((qt - qc)**2)
        sums["shoulder"] += np.sum(u[SHOULDER]**2)
        sums["elbow"]    += np.sum(u[ELBOW]**2)
        d_lo = np.maximum(qt - JOINT_LIMITS[:, 0] + eps, 1e-6)
        d_hi = np.maximum(JOINT_LIMITS[:, 1] - qt + eps, 1e-6)
        sums["jlimit"]   += np.sum(1/d_lo**2 + 1/d_hi**2)

    return {k: max(v / N, 1e-12) for k, v in sums.items()}









#  TRAJECTORY OPTIMIZER (L-BFGS-B)


def optimize_trajectory(q0, qf, ref, style):
    """Find optimal trajectory for a given style using L-BFGS-B."""
    N = CONFIG["solve_steps"]
    T = style["duration"]
    dt = T / N
    num_joints = 4

    
    comfort_pose = COMFORT_POSES[style["comfort_id"]]
    joint_limit_buffer = 0.05
    w_goal     = style["goal_weight"]    / ref["goal"]
    w_speed    = style["speed_weight"]   / ref["speed"]
    w_effort   = style["effort_weight"]  / ref["effort"]
    w_comfort  = style["comfort_weight"] / ref["comfort"]
    w_shoulder = style["shoulder_ratio"] / ref["shoulder"]
    w_elbow    = (1 - style["shoulder_ratio"]) / ref["elbow"]
    w_limit    = style["limit_weight"]   / ref["jlimit"]

    
    _, _, accel_init = min_jerk_trajectory(q0, qf, T, N)
    accel_init_flat = accel_init[:N].ravel()

    def rollout(accel_flat):
        accel = accel_flat.reshape(N, num_joints)
        vel = np.zeros((N + 1, num_joints))
        vel[1:] = np.cumsum(accel, axis=0) * dt
        pos = np.zeros((N + 1, num_joints))
        pos[0] = q0
        pos[1:] = q0 + np.cumsum(vel[:-1] * dt + 0.5 * accel * dt**2, axis=0)
        return pos, vel, accel

    def cost(accel_flat):
        pos, vel, accel = rollout(accel_flat)
        pos_steps = pos[1:]   # positions at each step 
        vel_steps = vel[1:]   # velocities at each step

        total_cost  = w_goal     * np.sum((pos_steps - qf)**2)
        total_cost += w_speed    * np.sum(vel_steps**2)
        total_cost += w_effort   * np.sum(accel**2)
        total_cost += w_comfort  * np.sum((pos_steps - comfort_pose)**2)
        total_cost += w_shoulder * np.sum(accel[:, SHOULDER]**2)
        total_cost += w_elbow    * np.sum(accel[:, ELBOW]**2)

        dist_to_lower = np.maximum(pos_steps - JOINT_LIMITS[:, 0] + joint_limit_buffer, 1e-6)
        dist_to_upper = np.maximum(JOINT_LIMITS[:, 1] - pos_steps + joint_limit_buffer, 1e-6)
        total_cost += w_limit * np.sum(1/dist_to_lower**2 + 1/dist_to_upper**2)
        total_cost *= dt

        # Terminal penalties: must reach goal and stop
        total_cost += 200 * np.sum((pos[-1] - qf)**2)
        total_cost += 100 * np.sum(vel[-1]**2)
        return total_cost

    result = minimize(cost, accel_init_flat, method="L-BFGS-B",
                      options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-7})
    pos, vel, _ = rollout(result.x)
    return pos, vel, result.fun







#  SAMPLE STYLES AND BUILD DEMO LIBRARY


def sample_styles(K, seed):
    
    rng = np.random.default_rng(seed)
    r = CONFIG["style_ranges"]
    styles = []
    ew  = lhs_samples(K, *r["effort_weight"], rng)
    sw  = lhs_samples(K, *r["speed_weight"], rng)
    lw  = lhs_samples(K, *r["limit_weight"], rng)
    sr  = lhs_samples(K, *r["shoulder_ratio"], rng)
    gw  = lhs_samples(K, *r["goal_weight"], rng)
    cw  = lhs_samples(K, *r["comfort_weight"], rng)
    dur = lhs_samples(K, *r["duration"], rng)
    cid = rng.integers(0, 3, size=K)

    for i in range(K):
        styles.append({
            "effort_weight":  float(ew[i]),
            "speed_weight":   float(sw[i]),
            "limit_weight":   float(lw[i]),
            "shoulder_ratio": float(sr[i]),
            "goal_weight":    float(gw[i]),
            "comfort_weight": float(cw[i]),
            "duration":       float(dur[i]),
            "comfort_id":     int(cid[i]),
        })
    return styles


def resample_to_phase(q, qd, N_phase):
    
    num_steps = len(q) - 1
    phase_grid_old = np.linspace(0, 1, num_steps + 1)
    phase_grid_new = np.linspace(0, 1, N_phase)
    q_resampled = np.zeros((N_phase, 4))
    qd_resampled = np.zeros((N_phase, 4))
    for j in range(4):
        q_resampled[:, j] = np.interp(phase_grid_new, phase_grid_old, q[:, j])
        qd_resampled[:, j] = np.interp(phase_grid_new, phase_grid_old, qd[:, j])
    return q_resampled, qd_resampled


def build_demo_library(q0, qf, ref, styles):
    
    import time
    N_phase = CONFIG["phase_points"]
    N_solve = CONFIG["solve_steps"]

    _, qd_ref, _ = min_jerk_trajectory(q0, qf, 2.0, N_solve)
    max_vel = 2.5 * np.max(np.abs(qd_ref))

    demos = []
    rej = {"goal": 0, "vel": 0, "fail": 0}
    t0 = time.time()

    for i, sty in enumerate(styles):
        try:
            q, qd, cost = optimize_trajectory(q0, qf, ref, sty)
        except Exception:
            rej["fail"] += 1
            continue

        if np.linalg.norm(q[-1] - qf) > 0.10:
            rej["goal"] += 1
            continue
        if np.max(np.abs(qd)) > max_vel:
            rej["vel"] += 1
            continue

        q_resampled, qd_resampled = resample_to_phase(q, qd, N_phase)
        demos.append({"q_phase": q_resampled, "T": sty["duration"]})

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(styles)}] valid={len(demos)} ({time.time()-t0:.0f}s)")

    print(f"    Done: {len(demos)} valid, rejected: {rej} ({time.time()-t0:.0f}s)")
    return demos









#  ProMP CLASS
class ProMP:
    

    def __init__(self):
        B = CONFIG["num_basis"]
        N = CONFIG["phase_points"]
        self.num_basis = B
        self.num_joints = 4
        self.s_grid = np.linspace(0, 1, N)

        centers = np.linspace(0, 1, B)
        spacing = centers[1] - centers[0] if B > 1 else 0.5
        width = spacing * CONFIG["basis_width"]

        # Precompute basis matrix at all phase points
        self.basis_matrix = np.zeros((N, B))
        for b in range(B):
            self.basis_matrix[:, b] = np.exp(-0.5 * ((self.s_grid - centers[b]) / width)**2)
        self.basis_matrix /= np.maximum(self.basis_matrix.sum(axis=1, keepdims=True), 1e-10)
        self.centers = centers
        self.width = width
        self.weight_mean = None
        self.weight_cov = None



    def _basis_at(self, s):
        
        phi = np.exp(-0.5 * ((s - self.centers) / self.width)**2)
        phi /= max(phi.sum(), 1e-10)
        return phi

    def _block_basis(self, Phi_single):
        
        M = Phi_single.shape[0]
        Phi_block = np.zeros((M * self.num_joints, self.num_basis * self.num_joints))
        for d in range(self.num_joints):
            Phi_block[d*M:(d+1)*M, d*self.num_basis:(d+1)*self.num_basis] = Phi_single
        return Phi_block

    def fit(self, demo_positions):
        
        K = len(demo_positions)
        Phi_block = self._block_basis(self.basis_matrix)

        W = np.zeros((K, self.num_basis * self.num_joints))
        for k in range(K):
            y = demo_positions[k].T.ravel()
            W[k] = np.linalg.lstsq(Phi_block, y, rcond=None)[0]

        self.weight_mean = np.mean(W, axis=0)
        if K > 1:
            diff = W - self.weight_mean
            self.weight_cov = (diff.T @ diff) / (K - 1)
        else:
            self.weight_cov = 0.01 * np.eye(self.num_basis * self.num_joints)
        self.weight_cov += 1e-6 * np.eye(self.num_basis * self.num_joints)

    def condition(self, s_obs, y_obs):
        
        sigma = CONFIG["obs_noise"]
        s_obs = np.atleast_1d(s_obs)
        y_obs = np.atleast_2d(y_obs)
        num_obs = len(s_obs)

        
        basis_at_obs = np.zeros((num_obs, self.num_basis))
        for m in range(num_obs):
            basis_at_obs[m] = self._basis_at(s_obs[m])

        obs_matrix = np.zeros((num_obs * self.num_joints, self.num_basis * self.num_joints))
        for d in range(self.num_joints):
            for m in range(num_obs):
                obs_matrix[m * self.num_joints + d, d*self.num_basis:(d+1)*self.num_basis] = basis_at_obs[m]

        obs_flat = y_obs.ravel()
        obs_noise_cov = sigma**2 * np.eye(num_obs * self.num_joints)

        innovation_cov = obs_matrix @ self.weight_cov @ obs_matrix.T + obs_noise_cov
        try:
            innovation_cov_inv = np.linalg.solve(innovation_cov, np.eye(innovation_cov.shape[0]))
        except np.linalg.LinAlgError:
            innovation_cov_inv = np.linalg.pinv(innovation_cov)

        kalman_gain = self.weight_cov @ obs_matrix.T @ innovation_cov_inv
        prediction_error = obs_flat - obs_matrix @ self.weight_mean

        mu_new = self.weight_mean + kalman_gain @ prediction_error
        Sigma_new = self.weight_cov - kalman_gain @ obs_matrix @ self.weight_cov
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
        return mu_new, Sigma_new

    def predict_at(self, s, mu_w=None, Sigma_w=None):
        """Predict joint positions at phase s. Returns (q, std)."""
        if mu_w is None: mu_w = self.weight_mean
        if Sigma_w is None: Sigma_w = self.weight_cov
        basis_vals = self._basis_at(s)
        q = np.zeros(self.num_joints)
        std = np.zeros(self.num_joints)
        for d in range(self.num_joints):
            joint_weights = mu_w[d*self.num_basis:(d+1)*self.num_basis]
            q[d] = basis_vals @ joint_weights
            joint_cov = Sigma_w[d*self.num_basis:(d+1)*self.num_basis, d*self.num_basis:(d+1)*self.num_basis]
            std[d] = np.sqrt(max(basis_vals @ joint_cov @ basis_vals, 0))
        return q, std










class OnlinePredictor:
    """Infers duration + predicts trajectory from streaming observations."""

    def __init__(self, promp):
        self.promp = promp
        self.duration_candidates = np.linspace(CONFIG["T_min"], CONFIG["T_max"],
                                               CONFIG["num_T_hypotheses"])
        self.log_duration_belief = np.full(len(self.duration_candidates),
                                           -np.log(len(self.duration_candidates)))

    @property
    def T_est(self):
        return float(self.duration_candidates[np.argmax(self.log_duration_belief)])

    def step(self, t_now, obs_window, dt):
        sigma = CONFIG["obs_noise"]
        _, q_now = obs_window[-1]

        # Update belief for each T hypothesis
        for k, Tk in enumerate(self.duration_candidates):
            sk = t_now / Tk
            if sk > 1.05:
                self.log_duration_belief[k] += -50
                continue
            sk = min(sk, 1.0)
            mu_k, std_k = self.promp.predict_at(sk)
            var_k = std_k**2 + sigma**2
            diff = q_now - mu_k
            ll = -0.5 * np.sum(diff**2 / var_k + np.log(2*np.pi*var_k))
            self.log_duration_belief[k] += ll

        self.log_duration_belief -= np.max(self.log_duration_belief)

        
        T_best = self.T_est
        s_list, y_list = [], []
        for t_i, q_i in obs_window:
            si = t_i / T_best
            if si <= 1.0:
                s_list.append(si)
                y_list.append(q_i)

        if s_list:
            mu_c, Sig_c = self.promp.condition(np.array(s_list),
                                                np.array(y_list))
        else:
            mu_c, Sig_c = self.promp.weight_mean, self.promp.weight_cov

        
        s_next = min((t_now + dt) / T_best, 1.0)
        q_pred, q_std = self.promp.predict_at(s_next, mu_c, Sig_c)

        return {"q_pred": q_pred, "q_std": q_std, "T_est": T_best,
                "mu_c": mu_c, "Sig_c": Sig_c}









#  RUN PREDICTION ON A TEST TRAJECTORY


def run_prediction(promp, q_true, qdot_true, dt, T_true):
    rng = np.random.default_rng(123)
    sigma = CONFIG["obs_noise"]
    q_noisy = q_true + rng.normal(0, sigma, q_true.shape)
    horizons = CONFIG["lookahead_horizons"]

    T_steps = len(q_true)
    win_size = CONFIG["condition_window"]
    engine = OnlinePredictor(promp)

    # Storage
    N = T_steps - 1
    preds = np.zeros((N, 4))
    vel_preds = np.zeros((N, 4))
    stds = np.zeros((N, 4))
    T_hist = np.zeros(N)
    preds_05 = np.zeros((N, 4))
    gt_05 = np.zeros((N, 4))
    valid_05 = np.full(N, False)

    # Lookahead storage
    horizon_frames = [max(1, int(round(h / dt))) for h in horizons]
    lookahead = {f"{h:.2f}s": {"promp": [], "linear": []}
                 for h in horizons}

    for n in range(N):
        t_n = n * dt

        
        start = max(0, n - win_size + 1)
        window = [(i * dt, q_noisy[i]) for i in range(start, n + 1)]

        
        out = engine.step(t_n, window, dt)
        preds[n] = out["q_pred"]
        stds[n] = out["q_std"]
        T_hist[n] = out["T_est"]

        # Velocity prediction via phase derivative
        T_best = out["T_est"]
        s_now = min(t_n / T_best, 1.0) if T_best > 0 else 1.0
        ds = 0.005
        s_fwd = min(s_now + ds, 1.0)
        s_bwd = max(s_now - ds, 0.0)
        q_fwd, _ = promp.predict_at(s_fwd, out["mu_c"], out["Sig_c"])
        q_bwd, _ = promp.predict_at(s_bwd, out["mu_c"], out["Sig_c"])
        gap = s_fwd - s_bwd
        if gap > 0 and T_best > 0:
            vel_preds[n] = (q_fwd - q_bwd) / (gap * T_best)
        else:
            vel_preds[n] = 0.0

        h_plot = 0.5
        h_plot_frames = int(round(h_plot / dt))
        target_plot = n + h_plot_frames
        if target_plot < T_steps:
            s_fut_plot = min((t_n + h_plot) / T_best, 1.0)
            q_fut_plot, _ = promp.predict_at(s_fut_plot, out["mu_c"], out["Sig_c"])
            preds_05[n] = q_fut_plot
            gt_05[n] = q_true[target_plot]
            valid_05[n] = True

        # Multi-horizon lookahead
        T_best = out["T_est"]
        for h_sec, h_fr in zip(horizons, horizon_frames):
            target = n + h_fr
            if target >= T_steps:
                continue
            key = f"{h_sec:.2f}s"

            # ProMP prediction at future phase
            s_fut = min((t_n + h_sec) / T_best, 1.0)
            q_fut, _ = promp.predict_at(s_fut, out["mu_c"], out["Sig_c"])
            promp_err = np.mean(np.abs(q_fut - q_true[target]))
            lookahead[key]["promp"].append(promp_err)

            # Linear extrapolation baseline
            if n > 0:
                q_lin = q_noisy[n] + (q_noisy[n] - q_noisy[n-1]) * h_fr
                lin_err = np.mean(np.abs(q_lin - q_true[target]))
                lookahead[key]["linear"].append(lin_err)

    return {
        "preds": preds, "vel_preds": vel_preds,
        "stds": stds, "T_hist": T_hist,
        "q_noisy": q_noisy,
        "lookahead": lookahead, "T_est_final": T_hist[-1],
        "preds_05": preds_05, "gt_05": gt_05, "valid_05": valid_05,
    }










#  PRINT RESULTS


def print_results(label, T_true, result, q_true, qdot_true, dt):
    num_frames = len(result["preds"])
    third      = num_frames // 3
    joint_names = ["sh_yaw", "sh_pitch", "sh_roll", "elbow"]

    pos_err = np.abs(result["preds"]     - q_true[1:])
    vel_err = np.abs(result["vel_preds"] - qdot_true[1:])

    print()
    print("=" * 60)
    print(" ", label)
    print("=" * 60)
    t_err = abs(T_true - result["T_est_final"])
    print("  True duration:      ", round(T_true, 2), "s")
    print("  Estimated duration: ", round(result["T_est_final"], 2), "s",
          "  (error:", round(t_err, 3), "s)")

    
    def print_error_table(title, err_array):
        print()
        print(" ", title)
        # header row
        header = "  Phase      "
        for name in joint_names:
            header += name.rjust(10)
        header += "      mean"
        print(header)
        print("  " + "-" * (len(header) - 2))

        # one row per phase segment
        segments = [
            ("early", err_array[:third]),
            ("mid",   err_array[third : 2 * third]),
            ("late",  err_array[2 * third :]),
            ("TOTAL", err_array),
        ]
        for seg_name, chunk in segments:
            avg = np.mean(chunk, axis=0)
            row = "  " + seg_name.ljust(10)
            for val in avg:
                row += str(round(float(val), 4)).rjust(10)
            row += str(round(float(avg.mean()), 4)).rjust(10)
            print(row)

    print_error_table("Position Error (radians)", pos_err)
    print_error_table("Velocity Error (rad/s)",   vel_err)




    
    def print_comparison_table(title, true_array, pred_array):
        print()
        print(" ", title)
        header = "  time      "
        for name in joint_names:
            header += ("GT_" + name).rjust(10) + ("P_" + name).rjust(10)
        print(header)
        print("  " + "-" * (len(header) - 2))

        sample_every   = max(1, int(0.5 / dt))
        sample_indices = list(range(0, num_frames, sample_every))
        for n in sample_indices:
            t    = round((n + 1) * dt, 2)
            gt   = true_array[n + 1]
            pred = pred_array[n]
            row  = "  " + (str(t) + "s").ljust(10)
            for j in range(4):
                row += str(round(float(gt[j]),   4)).rjust(10)
                row += str(round(float(pred[j]), 4)).rjust(10)
            print(row)

    print_comparison_table(
        "Position: Ground Truth vs Predicted  (sampled every 0.5 s)",
        q_true, result["preds"]
    )
    print_comparison_table(
        "Velocity: Ground Truth vs Predicted  (sampled every 0.5 s)",
        qdot_true, result["vel_preds"]
    )







#  PLOT: 4 joints, GT vs Predicted


def plot_joints(q_true, result, dt, label):
    

    names = ["Sh Yaw", "Sh Pitch", "Sh Roll", "Elbow"]
    mask = result["valid_05"]
    t_plot = np.arange(len(mask))[mask] * dt
    gt = result["gt_05"][mask]
    pred = result["preds_05"][mask]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    fig.suptitle(f"{label}  0.5s Lookahead Prediction", fontsize=13)

    for j in range(4):
        ax = axes[j]
        ax.plot(t_plot, gt[:, j], "b-", lw=1.5, label="GT (t+0.5s)")
        ax.plot(t_plot, pred[:, j], "r--", lw=1.0, label="Pred (t+0.5s)")
        ax.set_title(names[j], fontsize=10)
        ax.set_xlabel("time (s)")
        if j == 0:
            ax.set_ylabel("q (rad)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()








#  MAIN


def main():
    q0 = np.array(CONFIG["q_start"])
    qf = np.array(CONFIG["q_goal"])
    dt = CONFIG["dt"]
    K = CONFIG["num_demos"]

    print("=" * 60)
    print("  ProMP Arm Motion Prediction")
    print("=" * 60)
    print(f"  q_start = {q0}")
    print(f"  q_goal  = {qf}")
    print(f"  dt = {dt}s, demos = {K}")



    #  Step 1: Reference costs 
    print("\n  Computing reference costs")
    ref = compute_ref_costs(q0, qf)



    #  Step 2: Generate demo library 
    print(f"\n  Generating {K} demo trajectories (L-BFGS-B)")
    styles = sample_styles(K, CONFIG["demo_seed"])
    demos = build_demo_library(q0, qf, ref, styles)

    

    #  Step 3: Fit ProMP 
    print(f"\n  Fitting ProMP on {len(demos)} demos")
    promp = ProMP()
    promp.fit([d["q_phase"] for d in demos])
    print(f"  Weight dim: {promp.num_basis * promp.num_joints}")
    print(f"  Sigma trace: {np.trace(promp.weight_cov):.4f}")

    #  Step 4: Generate ground truth 
    T_test = CONFIG["test_duration"]
    M = int(math.floor(T_test / dt)) + 1
    q_true = np.zeros((M, 4))
    
    for i in range(M):
        s = min(i * dt / T_test, 1.0)
        q_true[i] = q0 + (qf - q0) * min_jerk(s)
    
    qdot_true = finite_diff_vel(q_true, dt)

    
    print(f"\n  Test: {T_test}s smooth min-jerk motion")
    print(f"  Demo T range: [{min(d['T'] for d in demos):.2f}, "f"{max(d['T'] for d in demos):.2f}]s")

    #  Step 5: Run prediction 
    print(f"\n  Running online prediction")
    result = run_prediction(promp, q_true, qdot_true, dt, T_test)

    #  Step 6: Results 
    print_results(f"Test: {T_test}s Min-Jerk", T_test, result,q_true, qdot_true, dt)



    #  Step 7: Plot 
    plot_joints(q_true, result, dt,f"{T_test}s Min-Jerk Motion")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
