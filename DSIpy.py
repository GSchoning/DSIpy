import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from scipy.linalg import solve
from scipy.optimize import minimize, least_squares
from scipy.special import logit, expit
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial
import warnings
import pickle
import time

# Suppress OptimizeWarning from scipy during CG convergence issues
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')

class DSISurrogate:
    """
    Implements Data Space Inversion (DSI) using a covariance-based surrogate.
    """
    def __init__(self, obs_pca_variance=1, pred_pca_variance=1,
                 svd_variance=0.9999,
                 obs_transform='log',  # Options: 'log', 'normal_score', 'none'
                 pred_transform='log',  # Options: 'log', 'logit', 'normal_score', 'none'
                 pred_bounds=None,
                 epsilon=1e-10, regularization=1e-9):
        
        self.obs_pca_variance = obs_pca_variance
        self.pred_pca_variance = pred_pca_variance
        self.svd_variance = svd_variance
        
        self.obs_transform = obs_transform
        self.pred_transform = pred_transform
        self.pred_bounds = pred_bounds
        self.epsilon = epsilon
        self.regularization = regularization
        
        if self.pred_transform == 'logit' and self.pred_bounds is None:
            raise ValueError("pred_bounds=[a, b] must be provided when pred_transform='logit'")
            
        self.obs_scaler_ = None
        self.pred_scaler_ = None
        self.obs_nst_ = None  # QuantileTransformer for Observations
        self.pred_nst_ = None # QuantileTransformer for Predictions
        
        self.pca_obs_ = None
        self.pca_pred_ = None
        self.surrogate_components_ = {}
        self.is_fitted_ = False

    def _transform_data(self, data, kind='obs', direction='forward'):
        """
        Unified handler for Log, Logit, and Normal Score transforms.
        """
        method = self.obs_transform if kind == 'obs' else self.pred_transform
        
        # --- FORWARD (Physical -> Feature) ---
        if direction == 'forward':
            if method == 'normal_score':
                # Lazy initialization of NST transformer
                if kind == 'obs':
                    if self.obs_nst_ is None:
                        self.obs_nst_ = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(data), 1000))
                        return self.obs_nst_.fit_transform(data)
                    return self.obs_nst_.transform(data)
                else:
                    if self.pred_nst_ is None:
                        self.pred_nst_ = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(data), 1000))
                        return self.pred_nst_.fit_transform(data)
                    return self.pred_nst_.transform(data)
            
            elif method == 'log':
                return np.log10(np.maximum(data, self.epsilon))
            
            elif method == 'logit':
                a, b = self.pred_bounds
                scaled = (data - a) / (b - a)
                return logit(np.clip(scaled, self.epsilon, 1 - self.epsilon))
            
            else:
                return data

        # --- INVERSE (Feature -> Physical) ---
        else:
            if method == 'normal_score':
                transformer = self.obs_nst_ if kind == 'obs' else self.pred_nst_
                return transformer.inverse_transform(data)
            
            elif method == 'log':
                return 10**data
            
            elif method == 'logit':
                a, b = self.pred_bounds
                return expit(data) * (b - a) + a
            
            else:
                return data

    def fit(self, obs_prior, pred_prior):
        """Builds the DSI surrogate model from the prior ensemble."""
        print("--- Fitting DSI Surrogate Model ---")
        
        # 1. Transform Data (Log / Logit / Normal Score)
        print(f"Transforming Observations ({self.obs_transform})...")
        X_full = self._transform_data(obs_prior, kind='obs', direction='forward')
        
        print(f"Transforming Predictions ({self.pred_transform})...")
        y_full = self._transform_data(pred_prior, kind='pred', direction='forward')

        # 2. Scaling (Z-Score)
        print("Scaling data...")
        self.obs_scaler_ = StandardScaler().fit(X_full)
        self.pred_scaler_ = StandardScaler().fit(y_full)
        X_full_scaled = self.obs_scaler_.transform(X_full)
        y_full_scaled = self.pred_scaler_.transform(y_full)
        
        # 3. PCA
        print("Applying PCA...")
        if self.obs_pca_variance < 1.0:
            self.pca_obs_ = PCA(n_components=self.obs_pca_variance).fit(X_full_scaled)
            X_full_pca = self.pca_obs_.transform(X_full_scaled)
            n_pcs_obs = self.pca_obs_.n_components_
            print(f"Input PCA: Reduced {X_full.shape[1]} dims to {n_pcs_obs} components.")
        else:
            self.pca_obs_ = None
            X_full_pca = X_full_scaled
            n_pcs_obs = X_full.shape[1]
            print("Input PCA: Skipped.")
            
        if self.pred_pca_variance < 1.0:
            self.pca_pred_ = PCA(n_components=self.pred_pca_variance).fit(y_full_scaled)
            y_full_pca = self.pca_pred_.transform(y_full_scaled)
            n_pcs_pred = self.pca_pred_.n_components_
            print(f"Output PCA: Reduced {y_full.shape[1]} dims to {n_pcs_pred} components.")
        else:
            self.pca_pred_ = None
            y_full_pca = y_full_scaled
            n_pcs_pred = y_full.shape[1]
            print("Output PCA: Skipped.")

        # 4. SVD on Joint Covariance
        combined_final = np.hstack((X_full_pca, y_full_pca))
        print("Building surrogate components (Covariance + SVD)...")
        
        if np.isnan(combined_final).any() or np.isinf(combined_final).any():
            raise ValueError("Input contains NaN/Inf after pre-processing.")
            
        mean_combined = np.mean(combined_final, axis=0)
        mean_o_final = mean_combined[:n_pcs_obs]
        mean_s_final = mean_combined[n_pcs_obs:]
        joint_cov = np.cov(combined_final.T)
        joint_cov_reg = joint_cov + np.identity(joint_cov.shape[0]) * self.regularization
        
        try:
            U, s_vals, Vt = np.linalg.svd(joint_cov_reg)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SVD failed: {e}")
            
        cumulative_variance = np.cumsum(s_vals / np.sum(s_vals))
        try:
            n_components_svd = np.where(cumulative_variance >= self.svd_variance)[0][0] + 1
        except IndexError:
            n_components_svd = len(s_vals)
            
        print(f"Surrogate SVD: Keeping {n_components_svd} components.")
        U_trunc = U[:, :n_components_svd]
        Sigma_sqrt_trunc = np.sqrt(np.diag(np.maximum(s_vals[:n_components_svd], 1e-13)))
        
        M_matrix = U_trunc @ Sigma_sqrt_trunc
        M_obs = M_matrix[:n_pcs_obs, :]
        M_pred = M_matrix[n_pcs_obs:, :]
        
        self.surrogate_components_ = {
            'mean_o': mean_o_final, 'mean_s': mean_s_final,
            'M_obs': M_obs, 'M_pred': M_pred,
            'n_components': n_components_svd
        }
        self.is_fitted_ = True
        return self

    def save(self, filename):
        """Saves the surrogate state to a pickle file."""
        if not self.is_fitted_:
            print("Warning: Saving an unfitted surrogate.")
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Surrogate saved to '{filename}'")

    @classmethod
    def load(cls, filename):
        """Loads a surrogate state from a pickle file."""
        instance = cls()
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        instance.__dict__.update(state)
        return instance

    # --- Solvers ---
    
    def _solve_map_analytical(self, h_target_scaled, noise_variance, x_prior_sample=None):
        n_comp = self.surrogate_components_['n_components']
        if x_prior_sample is None: x_prior_sample = np.zeros(n_comp)
        lhs = self.surrogate_components_['M_obs'].T @ self.surrogate_components_['M_obs'] + noise_variance * np.identity(n_comp)
        rhs = (self.surrogate_components_['M_obs'].T @ (h_target_scaled - self.surrogate_components_['mean_o'])) + (noise_variance * x_prior_sample)
        try: x_map = solve(lhs, rhs, assume_a='pos')
        except np.linalg.LinAlgError: x_map = x_prior_sample
        return x_map

    def _solve_map_cg(self, h_target_scaled, noise_variance, x_prior_sample=None):
        n_comp = self.surrogate_components_['n_components']
        if x_prior_sample is None: x_prior_sample = np.zeros(n_comp)
        inv_noise_var = 1.0 / noise_variance
        M_obs_cg = self.surrogate_components_['M_obs']
        mu_o_cg = self.surrogate_components_['mean_o']
        def obj(x): return 0.5 * (inv_noise_var * np.sum((h_target_scaled - (mu_o_cg + M_obs_cg @ x))**2) + np.sum((x - x_prior_sample)**2))
        def grad(x): return -inv_noise_var * M_obs_cg.T @ (h_target_scaled - (mu_o_cg + M_obs_cg @ x)) + (x - x_prior_sample)
        return minimize(fun=obj, x0=x_prior_sample, method='CG', jac=grad, options={'maxiter': 5000}).x

    def _solve_map_ls(self, h_target_scaled, noise_variance, x_prior_sample=None):
        n_comp = self.surrogate_components_['n_components']
        if x_prior_sample is None: x_prior_sample = np.zeros(n_comp)
        noise_std = np.sqrt(noise_variance)
        M_obs_ls = self.surrogate_components_['M_obs']
        mu_o_ls = self.surrogate_components_['mean_o']
        def res(x): return np.concatenate(((h_target_scaled - (mu_o_ls + M_obs_ls @ x)) / noise_std, x - x_prior_sample))
        return least_squares(fun=res, x0=x_prior_sample, method='lm').x

    def _solve_ensemble_smoother(self, h_target_final, obs_noise_var_processed, n_posterior_samples):
        """
        Solves for the posterior ensemble using a single-step Ensemble Smoother.
        Updated to handle vector noise (diagonal covariance).
        """
        M_obs, mean_o = self.surrogate_components_['M_obs'], self.surrogate_components_['mean_o']
        n_pcs_obs = M_obs.shape[0]
        
        # 1. Prior Ensemble in Latent Space
        X_prior = np.random.normal(0.0, 1.0, size=(self.surrogate_components_['n_components'], n_posterior_samples))
        
        # 2. Observation Noise Setup
        # Ensure standard deviation is a column vector for broadcasting
        obs_noise_std_vec = np.sqrt(obs_noise_var_processed).reshape(-1, 1) 
        
        # 3. Perturb Observations (D_obs)
        H_true = np.tile(h_target_final.reshape(-1, 1), (1, n_posterior_samples))
        # Generate noise: N(0, 1) * std_vec
        noise_perturbations = np.random.normal(0.0, 1.0, size=(n_pcs_obs, n_posterior_samples)) * obs_noise_std_vec
        H_noisy = H_true + noise_perturbations
        
        # 4. Noise Covariance Matrix (Diagonal)
        C_noise = np.diag(obs_noise_var_processed)
        
        # 5. Predictions (D_pred)
        O_pred = (M_obs @ X_prior) + mean_o.reshape(-1, 1)
        
        # 6. Kalman Gain Calculation
        X_prime = X_prior - X_prior.mean(axis=1, keepdims=True)
        O_prime = O_pred - O_pred.mean(axis=1, keepdims=True)
        
        C_oo = (O_prime @ O_prime.T) / (n_posterior_samples - 1)
        C_xo = (X_prime @ O_prime.T) / (n_posterior_samples - 1)
        
        # Inversion with fallback
        matrix_to_invert = C_oo + C_noise
        try: 
            K_T = solve(matrix_to_invert, C_xo.T, assume_a='pos')
        except np.linalg.LinAlgError: 
            K_T = np.linalg.pinv(matrix_to_invert) @ C_xo.T
            
        # 7. Update
        return (X_prior + K_T.T @ (H_noisy - O_pred)).T

    def _solve_ies(self, h_target_final, obs_noise_var_processed, n_posterior_samples, n_ies_iterations):
        """
        Iterative Ensemble Smoother (ES-MDA).
        Updated to handle vector noise (diagonal covariance).
        """
        print(f"\n--- Performing ES-MDA with {n_posterior_samples} members ---")
        M_obs, mean_o = self.surrogate_components_['M_obs'], self.surrogate_components_['mean_o']
        n_pcs_obs = M_obs.shape[0]
        
        # Initial Ensemble
        X_k = np.random.normal(0.0, 1.0, size=(self.surrogate_components_['n_components'], n_posterior_samples))
        
        # Inflation Factor
        alpha = n_ies_iterations
        
        # Noise Setup (Vectorized)
        obs_noise_std_vec = np.sqrt(obs_noise_var_processed).reshape(-1, 1)
        
        # Inflated Covariance Matrix (Diagonal)
        C_noise_inflated = np.diag(obs_noise_var_processed) * alpha
        
        H_true = np.tile(h_target_final.reshape(-1, 1), (1, n_posterior_samples))
        
        print(f"{'Iter':<5} {'Mean Phi':<15} {'Cumul Time':<15}")
        start = time.time()
        
        for i in range(n_ies_iterations):
            # 1. Forward run (Surrogate Prediction)
            O_k = (M_obs @ X_k) + mean_o.reshape(-1, 1)
            
            # 2. Objective Function (Phi)
            # Calculate residuals weighted by specific feature noise
            res = (H_true - O_k) / obs_noise_std_vec
            phi = np.mean(np.sum(res**2, axis=0))
            print(f"{i+1:<5} {phi:<15.4f} {time.time()-start:<15.4f}")
            
            # 3. Perturb Observations with Inflated Noise
            # Perturbation scale: std * sqrt(alpha)
            noise_perturbations = np.random.normal(0.0, 1.0, size=(n_pcs_obs, n_posterior_samples)) * obs_noise_std_vec * np.sqrt(alpha)
            H_noisy = H_true + noise_perturbations
            
            # 4. Update Step
            X_prime = X_k - X_k.mean(axis=1, keepdims=True)
            O_prime = O_k - O_k.mean(axis=1, keepdims=True)
            
            C_oo = (O_prime @ O_prime.T) / (n_posterior_samples - 1)
            C_xo = (X_prime @ O_prime.T) / (n_posterior_samples - 1)
            
            matrix_to_invert = C_oo + C_noise_inflated
            
            try: 
                K_T = solve(matrix_to_invert, C_xo.T, assume_a='pos')
            except np.linalg.LinAlgError: 
                K_T = np.linalg.pinv(matrix_to_invert) @ C_xo.T
            
            X_k = X_k + K_T.T @ (H_noisy - O_k)
            
        return X_k.T

    # --- Metrics & Predict ---
    
    def _calculate_calibration_metrics(self, x_mean, h_obs, var, h_target):
        metrics = {'chi2_red': np.nan, 'mean_rel_error_perc': np.nan}
        o_pred = self.surrogate_components_['mean_o'] + self.surrogate_components_['M_obs'] @ x_mean
        
        # Use inverse transform method
        o_scaled = self.pca_obs_.inverse_transform(o_pred.reshape(1,-1)) if self.pca_obs_ else o_pred.reshape(1,-1)
        o_trans = self.obs_scaler_.inverse_transform(o_scaled)
        o_orig = self._transform_data(o_trans, kind='obs', direction='inverse')
        
        safe_den = np.where(np.abs(h_obs.flatten()) < 1e-10, 1.0, np.abs(h_obs.flatten()))
        metrics['mean_rel_error_perc'] = np.mean(np.abs(o_orig.flatten() - h_obs.flatten()) / safe_den) * 100
        return metrics

    def _estimate_transformed_noise(self, h_obs_orig, obs_noise_std_phys, n_samples=1000):
        """
        Estimates noise variance in the transformed (feature) space by propagating
        physical noise through the transformation pipeline via Monte Carlo.
        """
        n_obs = h_obs_orig.shape[1]
        
        # 1. Generate noisy samples in PHYSICAL space
        noise_gen = np.random.normal(0.0, obs_noise_std_phys, size=(n_samples, n_obs))
        h_monte_carlo = h_obs_orig + noise_gen
        
        # 2. Transform these samples to FEATURE space
        h_mc_trans = self._transform_data(h_monte_carlo, kind='obs', direction='forward')
        
        # 3. Calculate variance in the transformed space
        noise_var_trans = np.var(h_mc_trans, axis=0).reshape(1, -1)
        return noise_var_trans

    def predict(self, h_observed, obs_noise_std, inversion_type='ies', solver='analytical', 
                n_posterior_samples=500, n_ies_iterations=3, gd_learning_rate=1e-7, return_ensemble=False):
        
        if not self.is_fitted_: raise RuntimeError("Fit first.")
        h_obs_orig = h_observed.copy().reshape(1, -1)
        
        # --- 1. Robust Noise Propagation ---
        if self.obs_transform in ['normal_score', 'logit']:
            print("Estimating transformed noise via Monte Carlo propagation...")
            noise_proc = self._estimate_transformed_noise(h_obs_orig, obs_noise_std)
            h_proc = self._transform_data(h_obs_orig, kind='obs', direction='forward')
        
        elif self.obs_transform == 'log':
            h_proc = np.log10(np.maximum(h_obs_orig, self.epsilon))
            noise_proc = (obs_noise_std.copy().reshape(1, -1) / (np.maximum(h_obs_orig, self.epsilon) * np.log(10)))**2
        
        else:
            h_proc = h_obs_orig
            noise_proc = obs_noise_std.copy().reshape(1, -1)**2 

        # Scale Data and Noise
        h_scaled = self.obs_scaler_.transform(h_proc)
        noise_scaled = noise_proc / self.obs_scaler_.var_
        
        # PCA Projection
        if self.pca_obs_:
            h_target = self.pca_obs_.transform(h_scaled).flatten()
            # Approximation: Average noise variance if using scalar solvers, or use full vector for ES
            avg_noise = np.mean(noise_scaled)
            noise_final = np.ones(self.pca_obs_.n_components_) * avg_noise
        else:
            h_target = h_scaled.flatten()
            avg_noise = np.mean(noise_scaled)
            noise_final = noise_scaled.flatten()

        # --- 2. Invert ---
        scalar_noise_var = np.mean(noise_final)

        if inversion_type == 'map':
            x_post = self._solve_map_analytical(h_target, scalar_noise_var).reshape(1, -1)
            
        elif inversion_type == 'rml':
            print(f"--- RML ({n_posterior_samples}) ---")
            noise_std_scaled = np.sqrt(noise_scaled)
            
            # Generate perturbations in Feature space, then Scale -> PCA
            noise_gen_feat = np.random.normal(0.0, np.sqrt(noise_proc), size=(n_posterior_samples, h_obs_orig.shape[1]))
            noisy_h_proc = h_proc + noise_gen_feat
            noisy_h_scaled = self.obs_scaler_.transform(noisy_h_proc)
            
            if self.pca_obs_:
                noisy_targets = self.pca_obs_.transform(noisy_h_scaled)
            else:
                noisy_targets = noisy_h_scaled
                
            n_comp = self.surrogate_components_['n_components']
            x_priors = np.random.normal(0.0, 1.0, size=(n_posterior_samples, n_comp))
            x_post_list = []
            solver_func = self._solve_map_analytical if solver == 'analytical' else (self._solve_map_cg if solver == 'cg' else self._solve_map_ls)
            
            for k in range(n_posterior_samples):
                x_post_list.append(solver_func(noisy_targets[k], scalar_noise_var, x_priors[k]))
            x_post = np.array(x_post_list)
            
        elif inversion_type == 'es':
            x_post = self._solve_ensemble_smoother(h_target, noise_final, n_posterior_samples)
            
        elif inversion_type == 'ies':
            x_post = self._solve_ies(h_target, noise_final, n_posterior_samples, n_ies_iterations)
        
        # --- 3. Predict & Inverse Transform ---
        metrics = self._calculate_calibration_metrics(np.mean(x_post, axis=0), h_obs_orig, scalar_noise_var, h_target)
        print(f"Calibration Error: {metrics['mean_rel_error_perc']:.2f}%")
        
        pred_pca = self.surrogate_components_['mean_s'] + (self.surrogate_components_['M_pred'] @ x_post.T).T
        pred_scaled = self.pca_pred_.inverse_transform(pred_pca) if self.pca_pred_ else pred_pca
        pred_trans = self.pred_scaler_.inverse_transform(pred_scaled)
        
        pred_final = self._transform_data(pred_trans, kind='pred', direction='inverse')
        
        mean, std = np.mean(pred_final, axis=0), np.std(pred_final, axis=0)
        return (mean, std, metrics, pred_final) if return_ensemble else (mean, std, metrics)

    # --- Bias Correction & Plotting ---
    
    def diagnose_surrogate_bias(self, obs_prior, pred_prior, indices_to_plot=None, n_samples=None, figsize=(10, 8)):
        if not self.is_fitted_: raise RuntimeError("Fit first.")
        
        if n_samples and n_samples < len(obs_prior):
            idx = np.random.choice(len(obs_prior), n_samples, replace=False)
            obs_subset, pred_subset = obs_prior[idx], pred_prior[idx]
        else:
            obs_subset, pred_subset = obs_prior, pred_prior

        # Invert on Prior
        h_proc = self._transform_data(obs_subset, kind='obs', direction='forward')
        h_scaled = self.obs_scaler_.transform(h_proc)
        h_target = self.pca_obs_.transform(h_scaled) if self.pca_obs_ else h_scaled
        
        M_obs = self.surrogate_components_['M_obs']
        M_pred = self.surrogate_components_['M_pred']
        mean_o = self.surrogate_components_['mean_o']
        mean_s = self.surrogate_components_['mean_s']
        reg = 1e-6
        lhs = M_obs.T @ M_obs + reg * np.identity(M_obs.shape[1])
        rhs = (h_target - mean_o) @ M_obs
        X_est = (solve(lhs, rhs.T, assume_a='pos')).T
        
        pred_pca = mean_s + (X_est @ M_pred.T)
        pred_scaled = self.pca_pred_.inverse_transform(pred_pca) if self.pca_pred_ else pred_pca
        pred_trans = self.pred_scaler_.inverse_transform(pred_scaled)
        pred_surr = self._transform_data(pred_trans, kind='pred', direction='inverse')
        
        # Plot
        if indices_to_plot is None: indices_to_plot = [0]
        n_plots = len(indices_to_plot)
        fig, axes = plt.subplots(int(np.ceil(n_plots/2)), 2, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for i, idx in enumerate(indices_to_plot):
            if idx >= pred_subset.shape[1]: continue
            ax = axes[i]
            y_true, y_surr = pred_subset[:, idx], pred_surr[:, idx]
            ax.scatter(y_true, y_surr, alpha=0.5)
            mn, mx = min(y_true.min(), y_surr.min()), max(y_true.max(), y_surr.max())
            
            # UPDATED: Solid line instead of dashed, as requested
            ax.plot([mn, mx], [mn, mx], color='black', linestyle='-', linewidth=1.5)
            
            ax.set_xlabel("True Physics"); ax.set_ylabel("Surrogate")
            ax.set_title(f"Variable {idx} (Corr: {np.corrcoef(y_true, y_surr)[0,1]:.2f})")
        plt.tight_layout(); plt.show()

    def apply_bias_correction(self, posterior_ensemble, obs_prior, pred_prior, method='auto', poly_order=2, auto_threshold=0.6, seed=None):
        if not self.is_fitted_: raise RuntimeError("Fit first.")
        print(f"--- Bias Correction ({method}) ---")
        if seed is not None: np.random.seed(seed)
        
        # 1. Re-run on Prior
        h_proc = self._transform_data(obs_prior, kind='obs', direction='forward')
        h_scaled = self.obs_scaler_.transform(h_proc)
        h_target = self.pca_obs_.transform(h_scaled) if self.pca_obs_ else h_scaled
        
        M_obs = self.surrogate_components_['M_obs']
        M_pred = self.surrogate_components_['M_pred']
        mean_o = self.surrogate_components_['mean_o']
        mean_s = self.surrogate_components_['mean_s']
        reg = 1e-6
        lhs = M_obs.T @ M_obs + reg * np.identity(M_obs.shape[1])
        rhs = (h_target - mean_o) @ M_obs
        X_est = (solve(lhs, rhs.T, assume_a='pos')).T
        
        pred_pca = mean_s + (X_est @ M_pred.T)
        pred_scaled = self.pca_pred_.inverse_transform(pred_pca) if self.pca_pred_ else pred_pca
        pred_trans = self.pred_scaler_.inverse_transform(pred_scaled)
        
        # Surrogate Prior in FEATURE SPACE (Transformed)
        prior_surr_trans = pred_trans 

        # 2. Transform Truth & Posterior to Feature Space
        y_true_trans = self._transform_data(pred_prior, kind='pred', direction='forward')
        y_post_trans = self._transform_data(posterior_ensemble, kind='pred', direction='forward')
        
        corrected_post_trans = np.zeros_like(y_post_trans)
        n_vars = y_true_trans.shape[1]
        
        for i in range(n_vars):
            y_t = y_true_trans[:, i]
            y_s = prior_surr_trans[:, i]
            y_p = y_post_trans[:, i]
            
            curr_method = method
            if method == 'auto':
                curr_method = 'quantile' if np.corrcoef(y_t, y_s)[0,1] >= auto_threshold else 'error_inflation'
            
            if curr_method == 'quantile':
                # Safe Quantile Mapping with Clamping
                sort_idx = np.argsort(y_s)
                f_quant = interp1d(y_s[sort_idx], np.sort(y_t), kind='linear', bounds_error=False, fill_value=(y_t.min(), y_t.max()))
                corrected_post_trans[:, i] = f_quant(y_p)
            elif curr_method == 'polynomial':
                try:
                    p = Polynomial.fit(y_s, y_t, deg=poly_order)
                    corrected_post_trans[:, i] = p(y_p)
                except: corrected_post_trans[:, i] = y_p
            elif curr_method == 'linear':
                try:
                    p = Polynomial.fit(y_s, y_t, deg=1)
                    corrected_post_trans[:, i] = p(y_p)
                except: corrected_post_trans[:, i] = y_p
            elif curr_method == 'error_inflation':
                err_std = np.std(y_t - y_s)
                corrected_post_trans[:, i] = y_p + np.random.normal(0, err_std, size=y_p.shape)
        
        # 3. Inverse Transform back to Original Space
        return self._transform_data(corrected_post_trans, kind='pred', direction='inverse')
