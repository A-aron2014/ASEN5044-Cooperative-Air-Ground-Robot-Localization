import sympy as sp
import numpy as np
from scipy.linalg import expm
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from scipy.stats import chi2
import matplotlib.pyplot as plt


def discretize_ct_system(A, B, C, dt):
    """
    Given continuous-time linearized matrices A, B, C and a timestep dt,
    return the discrete-time linearized system matrices F, G, H using ZOH matrix exponential
    """
    n = A.shape[0]
    m = B.shape[1]
    
    AB = np.hstack((A, B))
    A_hat = np.vstack((AB, np.zeros((m, n+m))))
    Z = expm(A_hat*dt)
    
    F = Z[:n, :n]
    G = Z[:n, n:]
    H = C.copy()
    return F, G, H


def discretize_ct_forward_euler(A, B, C, dt):
    n = A.shape[0]
    I = np.eye(n)
    
    # Forward-Euler discretization
    F = I + dt * A      # State transition matrix
    G = dt * B          # Input matrix
    H = C.copy()        # Measurement matrix (instantaneous, so same form)
    
    return F, G, H


def check_stability(F, tol=1e-9):
    """
    Print eigenvalues of F and classify stability (discrete-time).
    """
    eigvals = np.linalg.eigvals(F)
    mags = np.abs(eigvals)
    
    print("Eigenvalues of F:")
    for lam, mag in zip(eigvals, mags):
        print(f"  λ = {lam:.6f}, |λ| = {mag:.6f}")
    
    if np.all(mags < 1 - tol):
        print("\nStability: asymptotically stable (all |λ| < 1).")
    elif np.any(mags > 1 + tol):
        print("\nStability: unstable (some |λ| > 1).")
    else:
        print("\nStability: marginal (at least one |λ| ≈ 1, none clearly > 1).")


def observability_matrix(F, H):
    """
    Build the observability matrix:
        O = [ H
              H F
              H F^2
              ...
              H F^{n-1} ]
    where n is the state dimension.
    """
    n = F.shape[0]
    O_blocks = []
    F_power = np.eye(n)
    
    for _ in range(n):
        O_blocks.append(H @ F_power)
        F_power = F @ F_power
    
    return np.vstack(O_blocks)


def check_observability(F, H, tol=1e-9):
    """
    Compute and print the rank of the observability matrix.
    """
    O = observability_matrix(F, H)
    rank = np.linalg.matrix_rank(O, tol=tol)
    n = F.shape[0]
    
    print(f"Observability matrix shape: {O.shape}")
    print(f"Rank(O) = {rank} (state dimension = {n})")
    
    if rank == n:
        print("System is locally observable around the linearization point.")
    else:
        print("System is NOT fully observable around the linearization point.")


def simulate_truth_trajectory(
    Q, R,
    delta_x0_mean, P0_plus,
    x_traj_nom,     # only x_traj_nom[0] is used here
    u_nom, L_nom, dt,
    f_ct, h_meas, rng=None
):
    """
    Generate ONE nonlinear truth trajectory + noisy measurements for TMT.

    Returns:
      - x_true: (N_steps, n)
      - y_meas: (N_steps, p)
    """
    if rng is None:
        rng = np.random.default_rng()
    N_steps, n = x_traj_nom.shape
    p = R.shape[0]

    # ===============================================
    #     Truth simulation (nonlinear + noises)
    # ===============================================
    
    # Sample true initial perturbation and build true initial state
    delta_x0_true = rng.multivariate_normal(mean=delta_x0_mean, cov=P0_plus)
    x_true = np.zeros((N_steps, n))
    x_true[0, :] = x_traj_nom[0, :] + delta_x0_true
    w = rng.multivariate_normal(mean=np.zeros(n), cov=Q, size=N_steps)  # process noise
    for k in range(N_steps - 1):
        x_true[k+1, :] = x_true[k, :] + dt * f_ct(x_true[k, :], u_nom, L_nom) + w[k]
    
    # Simulate noisy measurements from nonlinear h(x) with R_true
    y_meas = np.zeros((N_steps, p))
    y_meas[0, :] = np.nan  # t=0 often unused / NaNs as in ydata
    v = rng.multivariate_normal(mean=np.zeros(p), cov=R, size=N_steps)  # measurement noise
    for k in range(1, N_steps):
        y_meas[k, :] = h_meas(x_true[k, :]) + v[k]

    return x_true, y_meas


def run_LKF_on_truth(
    Q_KF, R_KF,
    delta_x0_mean, P0_plus,
    x_traj_nom, y_traj_nom,
    F_list, H_list,
    x_true, y_meas
):
    """
    Run ONE truth-model test (TMT) + linearized KF pass.

    Returns a dict with everything needed for NEES/NIS and plotting:
      - x_true, y_meas: nonlinear truth state and noisy measurement trajectories
      - x_hat_hist: reconstructed full state estimates
      - e_hist: state estimation errors (x_true - x_hat)
      - P_hist: posterior covariances P_k^+
      - innov_hist: innovations ν_k
      - S_hist: innovation covariances S_k
      - y_pred_hist: predicted measurements from the filter
    """
    N_steps, n = x_traj_nom.shape
    p = R_KF.shape[0]
    
    
    # ============================================
    #   Linearized KF around nominal trajectory
    # ============================================
    
    # Initial filter state in delta_x coordinates
    delta_x_hat_plus = delta_x0_mean.copy()  # filter's initial mean
    P_plus = P0_plus.copy()
    
    # Storage
    delta_x_hat_hist = np.zeros((N_steps, n))
    P_hist = np.zeros((N_steps, n, n))
    x_hat_hist = np.zeros((N_steps, n))
    y_pred_hist = np.zeros((N_steps, p))
    innov_hist = np.zeros((N_steps, p))
    S_hist = np.zeros((N_steps, p, p))
    
    for k in range(N_steps):
        x_nom_k = x_traj_nom[k, :]
        F_k = F_list[k]
        H_k = H_list[k]
        
        # Time update (predict)
        delta_x_hat_minus = F_k @ delta_x_hat_plus
        P_minus = F_k @ P_plus @ F_k.T + Q_KF
        S_k = H_k @ P_minus @ H_k.T + R_KF
        K_k = P_minus @ H_k.T @ np.linalg.inv(S_k)
        
        # Get innovation term
        y_meas_k = y_meas[k, :]
        y_pred_k = y_traj_nom[k] + H_k @ delta_x_hat_minus
        innov_k = np.zeros(p) if k == 0 else y_meas_k - y_pred_k    # t=0 data is NaNs
        
        # Measurement update (KF)
        delta_x_hat_plus = delta_x_hat_minus + K_k @ innov_k
        P_plus = (np.eye(n) - K_k @ H_k) @ P_minus
    
        # Reconstruct full state estimate
        x_hat_k = x_nom_k + delta_x_hat_plus
    
        # Log everything
        delta_x_hat_hist[k, :] = delta_x_hat_plus
        P_hist[k, :, :] = P_plus
        x_hat_hist[k, :] = x_hat_k
        y_pred_hist[k, :] = y_pred_k
        innov_hist[k, :] = innov_k
        S_hist[k, :, :] = S_k
    
    # Save state estimation errors
    e_hist = x_true - x_hat_hist  # shape (N_steps, 6)
    sigma_hist = np.sqrt(np.stack([np.diag(P_hist[k, :, :]) for k in range(N_steps)], axis=0))  # (N_steps, 6)

    return {
        "x_hat_hist": x_hat_hist,
        "e_hist": e_hist,
        "P_hist": P_hist,
        "sigma_hist": sigma_hist,
        "y_pred_hist": y_pred_hist,
        "innov_hist": innov_hist,
        "S_hist": S_hist,
    }


def plot_single_run_results(
    tvec,
    x_true,
    y_meas,
    x_hat_hist,
    y_pred_hist,
    e_hist,
    sigma_hist,
    state_labels,
    meas_labels,
    filter_name="KF",
    save_prefix=None
):
    """
    Shared plotting for Parts 4(a) and 5(a).
    
    Produces three figures:
      1. Truth vs estimate (states)
      2. Noisy measurement vs predicted measurement
      3. State estimation errors with +/- 2 sigma bounds

    If save_prefix is provided, files are saved as:
        f"{save_prefix}_states.png"
        f"{save_prefix}_measurements.png"
        f"{save_prefix}_errors.png"
    """

    n = x_true.shape[1]
    p = y_meas.shape[1]
    
    state_labels = [
        r"$\xi_g$ [m]",
        r"$\eta_g$ [m]",
        r"$\theta_g$ [rad]",
        r"$\xi_a$ [m]",
        r"$\eta_a$ [m]",
        r"$\theta_a$ [rad]",
    ]
    
    # Writing this manually since the format given in .mat file doesnt work with python
    meas_labels = [
        r"$\gamma_{ag}$ [rad]",
        r"$\rho_{ga}$ [m]",
        r"$\gamma_{ga}$ [rad]",
        r"$\xi_a$ [m]",
        r"$\eta_a$ [m]",
    ]
    
    # State Trajectories
    fig1, axes1 = plt.subplots(3, 2, figsize=(12, 9))
    axes1 = axes1.ravel()

    for i in range(n):
        ax = axes1[i]
        ax.plot(tvec, x_true[:, i], label="Nonlinear truth", linewidth=1.8)
        ax.plot(tvec, x_hat_hist[:, i], "--", label=f"{filter_name} estimate", linewidth=1.8)
        ax.set_ylabel(state_labels[i])
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        if i == 0:
            ax.legend(loc="best")

    fig1.suptitle(f"{filter_name}: Truth vs Estimated States", fontsize=14)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_prefix is not None:
        fig1.savefig(f"{save_prefix}_states.png")

    # Measuements
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 9))
    axes2 = axes2.ravel()

    for i in range(p):
        ax = axes2[i]
        ax.plot(tvec, y_meas[:, i], label="Noisy measurement", linewidth=1.8)
        ax.plot(tvec, y_pred_hist[:, i], "--", label=f"{filter_name} predicted meas", linewidth=1.8)
        ax.set_ylabel(meas_labels[i])
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        if i == 0:
            ax.legend(loc="best")

    axes2[-1].axis("off")

    fig2.suptitle(f"{filter_name}: Measurements vs Predicted Measurements", fontsize=14)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_prefix is not None:
        fig2.savefig(f"{save_prefix}_measurements.png")

    # State Estimation Error +2 sigma
    fig3, axes3 = plt.subplots(3, 2, figsize=(12, 9))
    axes3 = axes3.ravel()

    for i in range(n):
        ax = axes3[i]
        ax.plot(tvec, e_hist[:, i], label="Estimation error", linewidth=1.5)
        ax.plot(tvec,  2.0 * sigma_hist[:, i], "k--", label=r"$+2\sigma$", linewidth=1.0)
        ax.plot(tvec, -2.0 * sigma_hist[:, i], "k--", label=r"$-2\sigma$", linewidth=1.0)
        ax.set_ylabel(state_labels[i])
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        if i == 0:
            ax.legend(loc="best")

    fig3.suptitle(rf"{filter_name}: Estimation Errors with $\pm2\sigma$ Bounds (Single Run)", fontsize=14)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_prefix is not None:
        fig3.savefig(f"{save_prefix}_errors.png")

    plt.show()


def run_mc_TMT(
    filter_runner,      # run_LKF_on_truth or run_EKF_on_truth
    Q_true, R_true,
    Q_KF, R_KF,
    delta_x0_mean, P0_plus,
    x_traj_nom, y_traj_nom,
    F_list, H_list,    # only used by LKF; EKF can ignore via wrapper
    u_nom, L_nom, dt,
    N_mc, alpha, 
    f_ct, h_meas,
    A_eval=None, C_eval=None, rng=None
):
    """ 
    Run Monte Carlo Truth Model Tests
    Returns NEES and NIS data. 
    filter_runner can either be the LKF or EKF function
    """
    if rng is None:
        rng = np.random.default_rng()
        
    N_steps, n = x_traj_nom.shape
    p = R_true.shape[0]

    NEES = np.zeros((N_mc, N_steps))
    NIS  = np.zeros((N_mc, N_steps))

    for i in range(N_mc):
        # Get truth
        x_true, y_meas = simulate_truth_trajectory(
            Q=Q_true,
            R=R_true,
            delta_x0_mean=delta_x0_mean,
            P0_plus=P0_plus,
            x_traj_nom=x_traj_nom,
            u_nom=u_nom,
            L_nom=L_nom,
            dt=dt,
            f_ct=f_ct,
            h_meas=h_meas,
            rng=rng
        )

        # Run chosen filter on this truth
        if filter_runner is run_LKF_on_truth:
            result = filter_runner(
                Q_KF=Q_KF, R_KF=R_KF,
                delta_x0_mean=delta_x0_mean,
                P0_plus=P0_plus,
                x_traj_nom=x_traj_nom,
                y_traj_nom=y_traj_nom,
                F_list=F_list,
                H_list=H_list,
                x_true=x_true,
                y_meas=y_meas
            )
        else:
            result = filter_runner(
                Q_KF=Q_KF, R_KF=R_KF,
                delta_x0_mean=delta_x0_mean,
                P0_plus=P0_plus,
                x_traj_nom=x_traj_nom,
                u_nom=u_nom,
                L_nom=L_nom,
                x_true=x_true,
                y_meas=y_meas,
                dt=dt,
                f_ct=f_ct,
                h_meas=h_meas,
                A_eval=A_eval,
                C_eval=C_eval,
            )

        e_hist     = result["e_hist"]
        P_hist     = result["P_hist"]
        innov_hist = result["innov_hist"]
        S_hist     = result["S_hist"]

        # NEES
        for k in range(N_steps):
            e_k = e_hist[k, :]
            P_k = P_hist[k, :, :]
            NEES[i, k] = e_k @ np.linalg.solve(P_k, e_k)

        # NIS (skip k=0)
        for k in range(1, N_steps):
            innov_k = innov_hist[k, :]
            S_k = S_hist[k, :, :]
            NIS[i, k] = innov_k @ np.linalg.solve(S_k, innov_k)

    return NEES, NIS


def plot_consistency_stat(
    tvec_plot,
    samples,          # shape (N_mc, N_steps_plot)
    dof,              # n for NEES, p for NIS
    alpha,
    quantity_name,    # e.g. "NEES (LKF)" or "NIS (EKF)"
    ylabel,           # e.g. "NEES" or "NIS"
    legend_loc="upper left"
):
    """
    Plot consistency statistic (NEES or NIS) vs time with chi-square bounds.

    tvec_plot : (N_steps_plot,) time array
    samples   : (N_mc, N_steps_plot) array of per-run stats
    dof       : degrees of freedom (n for NEES, p for NIS)
    alpha     : significance level (e.g. 0.05 for 95% CI)
    """
    N_mc, N_steps_plot = samples.shape

    # Sample mean over Monte Carlo runs
    stat_mean = np.mean(samples, axis=0)

    # Chi-square bounds for the mean
    df_total = N_mc * dof
    lower_bound = chi2.ppf(alpha / 2.0, df=df_total) / N_mc
    upper_bound = chi2.ppf(1.0 - alpha / 2.0, df=df_total) / N_mc

    print(f"{quantity_name} mean bounds at {100*(1-alpha):.1f}% CI:")
    print(f"  lower: {lower_bound:.3f}, upper: {upper_bound:.3f}")
    print(f"  expected mean = dof = {dof}")

    # ------------------- Plot -------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot a few transparent trajectories to show spread
    # for i in range(min(N_mc, 20)):
    #     ax.plot(tvec_plot, samples[i, :], color="0.8", linewidth=0.6, alpha=0.6)

    # Plot mean across MC runs
    ax.plot(tvec_plot, stat_mean, label=f"Mean {quantity_name}", linewidth=2)

    # Expected value and bounds
    ax.hlines(dof, tvec_plot[0], tvec_plot[-1],
              colors="g", linestyles="--",
              label=f"Expected mean = {dof}")
    ax.hlines(lower_bound, tvec_plot[0], tvec_plot[-1],
              colors="r", linestyles="--",
              label=f"Lower bound ({100*(1-alpha):.1f}% CI)")
    ax.hlines(upper_bound, tvec_plot[0], tvec_plot[-1],
              colors="r", linestyles="--",
              label=f"Upper bound ({100*(1-alpha):.1f}% CI)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{quantity_name} Consistency Test")
    ax.grid(True)
    ax.legend(loc=legend_loc)

    plt.tight_layout()
    plt.show()


def run_EKF_on_truth(
    Q_KF, R_KF,
    delta_x0_mean, P0_plus,
    x_traj_nom,
    u_nom, L_nom,
    x_true, y_meas,
    dt, f_ct, h_meas,
    A_eval, C_eval,
):
    """
    Run ONE EKF pass on a given truth trajectory and measurements.
    Same return structure as run_LKF_on_truth.
    """
    N_steps, n = x_traj_nom.shape
    p = R_KF.shape[0]

    # Initial filter state: nominal + mean perturbation
    x_hat_plus = x_traj_nom[0, :] + delta_x0_mean
    P_plus = P0_plus.copy()

    # Storage
    x_hat_hist = np.zeros((N_steps, n))
    P_hist = np.zeros((N_steps, n, n))
    y_pred_hist = np.zeros((N_steps, p))
    innov_hist = np.zeros((N_steps, p))
    S_hist = np.zeros((N_steps, p, p))

    # Initial conditions
    x_hat_hist[0, :] = x_hat_plus
    P_hist[0, :, :] = P_plus
    y_pred_hist[0, :] = np.nan
    innov_hist[0, :] = np.zeros(p)
    S_hist[0, :, :] = np.eye(p)

    for k in range(N_steps):
        # Time update (predict)
        x_hat_minus = x_hat_plus + dt * f_ct(x_hat_plus, u_nom, L_nom)
        # Get matrices for current iteration
        F_k = np.eye(n) + dt * np.array(A_eval(x_hat_plus, u_nom, L_nom), dtype=float)
        H_k = C_k = np.array(C_eval(x_hat_minus), dtype=float)
        # Continue prediction step
        P_minus = F_k @ P_plus @ F_k.T + Q_KF
        S_k = H_k @ P_minus @ H_k.T + R_KF
        K_k = P_minus @ H_k.T @ np.linalg.inv(S_k)

        # Get innovation term
        y_meas_k = y_meas[k, :]
        y_pred_k = h_meas(x_hat_minus)    # EKF uses estimate instead of nominal
        innov_k = np.zeros(p) if k == 0 else y_meas_k - y_pred_k    # t=0 data is NaNs

        # Measurement update
        x_hat_plus = x_hat_minus + K_k @ innov_k
        P_plus = (np.eye(n) - K_k @ H_k) @ P_minus

        # Log everything
        x_hat_hist[k, :] = x_hat_plus
        P_hist[k, :, :] = P_plus
        y_pred_hist[k, :] = y_pred_k
        innov_hist[k, :] = innov_k
        S_hist[k, :, :] = S_k

    # Errors and sigmas
    e_hist = x_true - x_hat_hist
    sigma_hist = np.sqrt(np.stack([np.diag(P_hist[k, :, :]) for k in range(N_steps)], axis=0))

    return {
        "x_hat_hist": x_hat_hist,
        "e_hist": e_hist,
        "P_hist": P_hist,
        "sigma_hist": sigma_hist,
        "y_pred_hist": y_pred_hist,
        "innov_hist": innov_hist,
        "S_hist": S_hist,
    }