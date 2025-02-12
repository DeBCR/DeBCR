import numpy as np
from tqdm import tqdm

def sample_gamma(gamma_vec):
    sample_t = np.random.randint(1, gamma_vec.shape[0])
    gamma_tm = gamma_vec[sample_t - 1]
    gamma_t = gamma_vec[sample_t]
    gamma = np.random.uniform(gamma_tm, gamma_t)
    return gamma

def variance_schedule(timesteps, schedule_type='linear', min_beta=1e-4, max_beta=2e-2):
    if schedule_type == 'linear':
        alpha_vec = 1 - np.linspace(min_beta, max_beta, timesteps)
        gamma_vec_t = np.cumprod(alpha_vec)
    elif schedule_type == 'cos':
        t = np.arange(timesteps)
        s = 0.008
        gamma_vec_t = np.cos(((t / t[-1] + s) / (1 + s)) * np.pi / 2) ** 2
        gamma_vec_t /= gamma_vec_t[0]
        alpha_vec = np.zeros_like(gamma_vec_t)
        alpha_vec[0] = gamma_vec_t[0]
        for i in range(1, alpha_vec.shape[-1]):
            alpha_vec[i] = gamma_vec_t[i] / gamma_vec_t[i - 1]
        beta_vec = np.clip(1 - alpha_vec, 0, 0.999)
        alpha_vec = 1 - beta_vec

    return gamma_vec_t, alpha_vec


def weiner_sequences(n_paths, n_timesteps, img_size):
    noise_mat = np.zeros((n_paths, n_timesteps, img_size, img_size))
    for i in range(1, n_paths):
        for j in range(1, n_timesteps):
            noise_mat[i, j] = noise_mat[i, j - 1] + np.random.normal(0, 1, (img_size, img_size))
    return noise_mat
