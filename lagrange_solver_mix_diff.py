import torch
import torch.nn.functional as F
import math
import numpy as np

class LagrangeSolverMix:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type=None, # deprecated, not used
            correcting_x0_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995
    ):

        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return (noise, x0)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)
    
    def get_kernel_matrix(self, lambdas):
        return torch.vander(lambdas, N=len(lambdas), increasing=True)

    def ell(self, a, k, lam0, lam1):
        from math import exp, factorial
        
        if a == 0:
            return (lam1**(k+1) - lam0**(k+1)) / (k+1)
        
        F = lambda x: exp(a*x)*sum(
            ((-1)**m * factorial(k) / (factorial(k-m)*a**(m+1)))*x**(k-m)
            for m in range(k+1)
        )
        
        return F(lam1) - F(lam0)

    def get_integral_vector(self, a, lambda_s, lambda_t, lambdas):
        vector = [self.ell(a, k, lambda_s, lambda_t) for k in range(len(lambdas))]
        return torch.Tensor(vector, device=lambdas.device)

    def get_coefficients(self, tau, lambda_s, lambda_t, lambdas):
        # (p,)
        noise_integral = self.get_integral_vector(tau-1, lambda_s, lambda_t, lambdas)
        data_integral = self.get_integral_vector(tau, lambda_s, lambda_t, lambdas)
        
        # (p, p)
        kernel = self.get_kernel_matrix(lambdas)
        kernel_inv = torch.linalg.inv(kernel)
        
        # (p,)
        noise_coefficients = kernel_inv.T @ noise_integral
        data_coefficients = kernel_inv.T @ data_integral
        return noise_coefficients, data_coefficients

    def get_next_sample(self, sample, i, tau, hist, signal_rates, noise_rates, lambdas, p, corrector=False):
        '''
        sample : (b, c, h, w), tensor
        i : current sampling step, scalar
        tau : scalar
        hist : [ε_0, ε_1, ...] or [x_0, x_1, ...], tensor list
        signal_rates : [α_0, α_1, ...], tensor list
        lambdas : [λ_0, λ_1, ...], scalar list
        corrector : True or False
        '''
        
        # for predictor, (λ_i, λ_i-1, ..., λ_i-p+1), shape : (p,),
        # for corrector, (λ_i+1, λ_i, ..., λ_i-p+1), shape : (p+1,)
        lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])

        # for predictor, (c_i, c_i-1, ..., c_i-p+1), shape : (p,),
        # for corrector, (c_i+1, c_i, ..., c_i-p+1), shape : (p+1,)
        noise_coeffs, data_coeffs = self.get_coefficients(tau, lambdas[i], lambdas[i+1], lambda_array)
        
        # for predictor, (ε_i, ε_i-1, ..., ε_i-p+1), shape : (p,),
        # for corrector, (ε_i+1, λ_i, ..., ε_i-p+1), shape : (p+1,)
        hist = hist[i-p+1:i+(2 if corrector else 1)][::-1]
        hist_sum = sum([(tau-1)*nc*h[0] + tau*dc*h[1] for nc, dc, h in zip(noise_coeffs, data_coeffs, hist)])
        
        sample_coeff = (signal_rates[i+1]/signal_rates[i])**(1-tau) * (noise_rates[i+1]/noise_rates[i])**tau
        hist_coeff = signal_rates[i+1]**(1-tau) * noise_rates[i+1]**tau
        next_sample = sample_coeff*sample + hist_coeff*hist_sum
        return next_sample
    
    def sample(self, x, steps, taus, skip_type='logSNR', order=3):
        # tau : 0 = noise prediction, 1 = data prediction
        
        lower_order_final = True  # 전체 스텝이 매우 작을 때 마지막 스텝에서 차수를 낮춰서 안정성 확보할지.

        # 샘플링할 시간 범위 설정 (t_0, t_T)
        # diffusion 모델의 경우 t=1(혹은 T)에서 x는 가우시안 노이즈 상태라고 가정.
        t_0 = 1.0 / self.noise_schedule.total_N
        t_T = self.noise_schedule.T
        assert t_0 > 0 and t_T > 0, "Time range( t_0, t_T )는 0보다 커야 함. (Discrete DPMs: [1/N, 1])"

        # 텐서가 올라갈 디바이스 설정
        device = x.device

        # 샘플링 과정에서 gradient 계산은 하지 않으므로 no_grad()
        with torch.no_grad():

            # 실제로 사용할 time step array를 구한다.
            # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            lambdas = torch.Tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps])
            signal_rates = torch.Tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps])
            noise_rates = torch.Tensor([self.noise_schedule.marginal_std(t) for t in timesteps])
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            
            for i in range(0, steps):
                tau = taus[i]
                
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)
                    
                # ===predictor===
                x_pred = self.get_next_sample(x, i, tau, hist, signal_rates, noise_rates, lambdas,
                                              p=p, corrector=False)
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                x_corr = self.get_next_sample(x, i, tau, hist, signal_rates, noise_rates, lambdas,
                                              p=p, corrector=True)
                x = x_corr

        return x
        