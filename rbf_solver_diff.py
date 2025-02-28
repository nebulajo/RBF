import torch
import torch.nn.functional as F
import math
import numpy as np
from .samplers import expand_dims

class RBFSolver:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type="data_prediction",
            correcting_x0_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995
    ):

        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.predict_x0 = algorithm_type == "data_prediction"

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

        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

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
    
    def get_kernel_matrix(self, lambdas, width):
        # (p, 1)
        lambdas = lambdas[:, None]
        # (p, p)
        K = torch.exp(-width**2 * (lambdas - lambdas.T) ** 2)
        return K

#     def get_integral_vector(self, lambda_s, lambda_t, lambdas, width):
#         if width == 0:
#             return (torch.exp(-lambda_s) - torch.exp(-lambda_t)) * torch.ones_like(lambdas)
        
#         factor = torch.sqrt(torch.tensor(math.pi)) / (2*width)
#         exponent = torch.exp(-lambdas + 1.0/(4.0*width**2))
#         upper = torch.erf(width*(lambda_t - lambdas) + 1.0/(2.0*width))
#         lower = torch.erf(width*(lambda_s - lambdas) + 1.0/(2.0*width))
#         return factor * exponent * (upper - lower)
    
    def get_integral_vector(self, lambda_s, lambda_t, lambdas, width):
        import numpy as np
        from scipy.special import erf
        import math

        """
        Computes the integral vector equivalent to the PyTorch version,
        but using NumPy / SciPy operations.
        """
        # Handle the zero-width case.
        if width == 0:
            if self.predict_x0:
                return (np.exp(lambda_s) - np.exp(lambda_t)) * np.ones_like(lambdas)
            else:    
                return (np.exp(-lambda_s) - np.exp(-lambda_t)) * np.ones_like(lambdas)

        # Compute factor, exponent, and (upper - lower) via the error function.
        factor = np.sqrt(math.pi) / (2 * width)
        
        if self.predict_x0:
            exponent = np.exp(lambdas + 1.0 / (4.0 * width**2))
            upper = erf(width * (lambda_t - lambdas) - 1.0 / (2.0 * width))
            lower = erf(width * (lambda_s - lambdas) - 1.0 / (2.0 * width))
        else:
            exponent = np.exp(-lambdas + 1.0 / (4.0 * width**2))
            upper = erf(width * (lambda_t - lambdas) + 1.0 / (2.0 * width))
            lower = erf(width * (lambda_s - lambdas) + 1.0 / (2.0 * width))
            
        return factor * exponent * (upper - lower)

    def get_coefficients(self, lambda_s, lambda_t, lambdas, width):
        # (p,)
        integral = self.get_integral_vector(lambda_s, lambda_t, lambdas, width)
        # (p, p)
        kernel = self.get_kernel_matrix(lambdas, width)
        # (p,)
        coefficients = (integral[None, :] @ torch.linalg.inv(kernel))[0]
        return coefficients

    def get_next_sample(self, sample, i, hist, signal_rates, noise_rates, lambdas, p, width, corrector=False):
        '''
        sample : (b, c, h, w), tensor
        i : current sampling step, scalar
        hist : [ε_0, ε_1, ...] or [x_0, x_1, ...], tensor list
        signal_rates : [α_0, α_1, ...], tensor list
        lambdas : [λ_0, λ_1, ...], scalar list
        width : width of RBF kernel
        corrector : True or False
        '''
        
        # for predictor, (λ_i, λ_i-1, ..., λ_i-p+1), shape : (p,),
        # for corrector, (λ_i+1, λ_i, ..., λ_i-p+1), shape : (p+1,)
        lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])

        # for predictor, (c_i, c_i-1, ..., c_i-p+1), shape : (p,),
        # for corrector, (c_i+1, c_i, ..., c_i-p+1), shape : (p+1,)
        coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array, width)
#         if self.predict_x0:
#             print('coeffs sum :', torch.sum(coeffs), 'true sum :', torch.exp(lambdas[i+1]) - torch.exp(lambdas[i]))
#         else:
#             print('coeffs sum :', torch.sum(coeffs), 'true sum :', -torch.exp(-lambdas[i+1]) - (-torch.exp(-lambdas[i])))
        
        # for predictor, (ε_i, ε_i-1, ..., ε_i-p+1), shape : (p,),
        # for corrector, (ε_i+1, λ_i, ..., ε_i-p+1), shape : (p+1,)
        datas = hist[i-p+1:i+(2 if corrector else 1)][::-1]
        
        data_sum = sum([coeff * data for coeff, data in zip(coeffs, datas)])
        if self.predict_x0:
            next_sample = noise_rates[i+1]/noise_rates[i]*sample + noise_rates[i+1]*data_sum
        else:
            next_sample = signal_rates[i+1]/signal_rates[i]*sample - signal_rates[i+1]*data_sum
        return next_sample
    
    def sample(self, x, steps, skip_type='logSNR', order=3, gamma=3):
        
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
            width = steps / (gamma * abs(lambdas[-1] - lambdas[0]))
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            
            for i in range(0, steps):
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)
                    
                # ===predictor===
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, width=width, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, width=width, corrector=True)
                x = x_corr
        # 최종적으로 x를 반환
        return x
