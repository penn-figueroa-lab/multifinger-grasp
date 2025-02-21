import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import open3d as o3d
import imageio

def plot_errors(errors):
    
    errors = np.array(errors)

    R_err = errors[:, 0]  # rotation error 
    s_err = errors[:, 1]  # scale error
    t_err = errors[:, 2]  # translation error
    c_err = errors[:, 3]  # shape error

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(R_err, marker='o', label='Rotation Error')
    axes[0, 0].set_title('Rotation Error')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Error (radians)')
    # axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True)

    axes[0, 1].plot(s_err, marker='o', color='orange', label='Scale Error')
    axes[0, 1].set_title('Scale Error')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Absolute Difference')
    axes[0, 0].set_ylim(0, 10)
    axes[0, 1].grid(True)

    axes[1, 0].plot(t_err, marker='o', color='green', label='Translation Error')
    axes[1, 0].set_title('Translation Error')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('L2 Distance')
    axes[0, 0].set_ylim(0, 1)
    axes[1, 0].grid(True)

    axes[1, 1].plot(c_err, marker='o', color='red', label='Shape Error')
    axes[1, 1].set_title('Shape Error')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('L2 Distance')
    axes[0, 0].set_ylim(0, 1)
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def eval(R, s, t, c, R_est, s_est, t_est, c_est):
    R_diff = R.T @ R_est  # or R @ R_est.T
    trace_val = np.trace(R_diff)
    cos_theta = max(min((trace_val - 1.0) / 2.0, 1.0), -1.0)
    theta = np.arccos(cos_theta)  # in radians
    rotation_error = theta
    scale_error = abs(s - s_est)
    translation_error = np.linalg.norm(t - t_est)
    shape_error = np.linalg.norm(c_est - c)

    return rotation_error, scale_error, translation_error, shape_error

def run_admm(B0, B, Yt, c_gt, c_low, c_high, c_mean, ev, s, R, t, temp_t, Y, rho=1.0, iterations=10):

    Rt_pred = np.eye(3)
    q_pred = np.zeros((3, 1))
    a_pred = 1.

    Rt_nu = np.zeros((3, 3)) 
    q_nu = np.zeros((3, 1))
    
    # Yt 3 x n
    # B0 3 x n
    # B  3 x n x f
    # c  f

    # print("Yt", Yt.shape)
    # print("BO", B0.shape)
    # print("B", B.shape)
    n = B.shape[1]
    f = B.shape[2]

    errors = []

    for iteration in range(iterations):
        W = cp.Variable((3, 3))
        b = cp.Variable((3, 1))
        c = cp.Variable(f)

        Bc = cp.reshape(B.reshape(3*n, f) @ c, shape=(3, n), order='C')
        # print("eigenvalues", ev)
        P = np.diag(1 / np.array(ev))
        # P = np.diag(ev)

        # print("Bc", Bc.shape)
        cost = cp.sum_squares(W @ Yt - (B0 + Bc) - b) + \
                0.001 * cp.quad_form(c - c_mean, P) + \
               (rho / 2) * cp.sum_squares(a_pred * Rt_pred - W + Rt_nu) + \
               (rho / 2) * cp.sum_squares(q_pred - b + q_nu)
        #    0.001 * cp.sum_squares(c - c_mean) + \ 
         # 0.00001 * cp.quad_form(c - c_mean, P) + \

        constraints = [
            c <= c_high,
            c >= c_low,
            # B0 + Bc >= -0.5,
            # B0 + Bc <= 0.5
        ]
        prob = cp.Problem(cp.Minimize(cost), constraints=constraints)
        prob.solve()
        W = W.value
        b = b.value
        c = c.value

        U, _, Vt = np.linalg.svd(W)
        Rt_pred = U @ Vt
        if np.linalg.det(Rt_pred) < 0:
            Vt[-1, :] *= -1
            Rt_pred = U @ Vt
        a_pred = np.array([max(1e-12, np.trace(W.T @ Rt_pred) / np.trace(Rt_pred.T @ Rt_pred))])
        q_pred = b - q_nu

        Rt_nu = Rt_nu + a_pred * Rt_pred - W
        q_nu = q_nu + q_pred - b

        # Test convergence
        R_pred = Rt_pred.T
        s_pred = 1 / a_pred
        t_pred = (Rt_pred.T @ q_pred / a_pred)[:, 0]
        c_pred = c

        R_error, s_error, t_error, c_error = eval(R, s, t, c_gt, R_pred, s_pred[0], t_pred + temp_t[:, 0], c_pred)
        errors.append(np.array([R_error, s_error, t_error, c_error]))

        Xt = s_pred * R_pred @ (B0 + B @ c) + t_pred[:, None]

        difference = np.mean(np.linalg.norm(Yt - Xt, axis=0))
        # print("R_pred", R_pred)
        # print("s_pred", s_pred)
        # print("t_pred", t_pred)
        # print("c_pred", c_pred)
        print(f"difference between Yt and Xt: {difference:.6f}" )

    R_pred = Rt_pred.T
    s_pred = 1 / a_pred
    t_pred = (Rt_pred.T @ q_pred / a_pred)[:, 0]
    c_pred = c

    Xt = s_pred * R_pred @ (B0 + B @ c) + t_pred[:, None]

    return Xt, R_pred, s_pred, t_pred, c_pred, errors



