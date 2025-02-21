def get_uniform_timesteps(start=1, end=0, n_timesteps=10, expanded=False):
    timesteps = np.linspace(start, end, n_timesteps+1)
    current_timesteps = timesteps if expanded else timesteps[:-1]
    next_timesteps = timesteps[1:]
    return current_timesteps, next_timesteps

def get_integral_vector(a, b, s, beta):
    if beta == 0:
        return (b - a) * np.ones_like(s, dtype=float)
    
    return (np.sqrt(np.pi) / (2.0 * beta)) * (
        erf(beta * (b - s)) - erf(beta * (a - s))
    )

def get_kernel_matrix(rev_times, beta):
    if beta == 0:
        return np.ones((len(rev_times), len(rev_times)))
    
    # (p, 1)
    k = rev_times[:, None]
    # (p, p)
    K = np.exp(-beta**2 * (k-k.T)**2)
    return K

def get_next_sample(i, p, sample,
                    timesteps, next_timesteps, v_hist, beta, corrector=False):
    t_i = timesteps[i]
    t_i1 = next_timesteps[i]
    
    vector = 0.0
    if not corrector:
        rev_times = np.array([timesteps[i-j] for j in range(p)])
        # (p, p)
        K = get_kernel_matrix(rev_times, beta)
        # (p,)
        L = get_integral_vector(t_i, t_i1, rev_times, beta)
        # (p,)
        coeffs = np.linalg.inv(K) @ L
        for j, coeff in enumerate(coeffs):
            vector = vector + coeff * v_hist[i-j]    
    else:
        rev_times = np.array([timesteps[i+1-j] for j in range(p+1)])
        # (p, p)
        K = get_kernel_matrix(rev_times, beta)
        # (p,)
        L = get_integral_vector(t_i, t_i1, rev_times, beta)
        # (p,)
        coeffs = np.linalg.inv(K) @ L
        for j, coeff in enumerate(coeffs):
            vector = vector + coeff * v_hist[i+1-j]    
        
    next_sample = sample + vector
    return next_sample

def sample_from_model_RBF(
    generator,
    model,
    x_0: torch.Tensor,
    num_steps=20, p_order=2, p_only=False, beta=1
):
    NFE = 0
    timesteps, next_timesteps = get_uniform_timesteps(n_timesteps=num_steps)
    x_i = x_0.clone()
    v_hist = [None for _ in range(num_steps)]
    v_hist[0] = model(x_i, timesteps[0])
    NFE += 1
    with torch.no_grad():
        for i in range(0, num_steps):
            p = min(i+1, p_order)

            # ---- Predictor ----
            x_pred = get_next_sample(i, p, x_i, timesteps, next_timesteps, v_hist, beta)
            if i == num_steps - 1:
                x_i = x_pred
                break

            v_hist[i+1] = model(x_pred, next_timesteps[i])
            NFE += 1

            if p_only:
                x_i = x_pred
                continue

            # ---- Corrector ----
            x_corr = get_next_sample(i, p, x_i, timesteps, next_timesteps, v_hist, beta, corrector=True)
            x_i = x_corr

    return NFE, x_i