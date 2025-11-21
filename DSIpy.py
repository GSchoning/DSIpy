import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import solve
from scipy.optimize import minimize, least_squares
from scipy.special import logit, expit
from numpy.polynomial import Polynomial
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster, get_client
import warnings
import pickle
import time

# Suppress OptimizeWarning from scipy during CG convergence issues
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

class DSISurrogate:
    """
    Implements Data Space Inversion (DSI) using a covariance-based surrogate.

    Builds a surrogate linking observations and predictions based on a prior
    ensemble and allows conditioning using MAP, RML, ES, or IES (ES-MDA).

    Args:
        obs_pca_variance (float): Variance threshold for input PCA (e.g., 0.95).
                                  Set >= 1.0 to skip input PCA.
        pred_pca_variance (float): Variance threshold for output PCA (e.g., 0.99).
                                   Set >= 1.0 to skip output PCA.
        svd_variance (float): Variance threshold for surrogate SVD (e.g., 0.999).
        log_transform_obs (bool): If True, applies log10 transform to observations.
        pred_transform (str): Transform to apply to predictions:
                              'log' -> log10(y) for [0, +inf) bounds.
                              'logit' -> logit((y-a)/(b-a)) for [a, b] bounds.
                              'none' -> No transform.
        pred_bounds (list or tuple): Required if pred_transform='logit'. E.g., [0, 1].
        epsilon (float): Small value to add/clip to avoid log(0) or logit(0/1).
        regularization (float): Small value added to covariance diagonal for SVD stability.
    """
    def __init__(self, obs_pca_variance=1, pred_pca_variance=1,
                 svd_variance=0.9999,
                 log_transform_obs=True, pred_transform='log',
                 pred_bounds=None,
                 epsilon=1e-10, regularization=1e-9):
        
        self.obs_pca_variance = obs_pca_variance
        self.pred_pca_variance = pred_pca_variance
        self.svd_variance = svd_variance
        self.log_transform_obs = log_transform_obs
        self.pred_transform = pred_transform
        self.pred_bounds = pred_bounds
        self.epsilon = epsilon
        self.regularization = regularization
        
        if self.pred_transform == 'logit' and self.pred_bounds is None:
            raise ValueError("pred_bounds=[a, b] must be provided when pred_transform='logit'")
            
        self.obs_scaler_ = None
        self.pred_scaler_ = None
        self.pca_obs_ = None
        self.pca_pred_ = None
        self.surrogate_components_ = {}
        self.is_fitted_ = False

    def fit(self, obs_prior, pred_prior):
        """Builds the DSI surrogate model from the prior ensemble."""
        print("--- Fitting DSI Surrogate Model ---")
        X_full = obs_prior.copy()
        y_full = pred_prior.copy()

        # --- 1a. Apply Log Transform (Observations) ---
        if self.log_transform_obs:
            print("Applying log10 transform to observations...")
            X_full = np.log10(np.maximum(X_full, self.epsilon))
            
        # --- 1b. Apply Transform (Predictions) ---
        if self.pred_transform == 'log':
            print("Applying log10 transform to predictions...")
            y_full = np.log10(np.maximum(y_full, self.epsilon))
        elif self.pred_transform == 'logit':
            print(f"Applying logit transform to predictions with bounds {self.pred_bounds}...")
            a, b = self.pred_bounds
            if a >= b:
                raise ValueError(f"Invalid pred_bounds: {self.pred_bounds}. Must be [min, max].")
            
            # --- Optimization: Bounds Check ---
            if np.min(y_full) < a or np.max(y_full) > b:
                n_bad = np.sum((y_full < a) | (y_full > b))
                print(f"Warning: {n_bad} prior predictions are outside bounds [{a}, {b}]. They will be clipped.")

            y_scaled_01 = (y_full - a) / (b - a)
            y_full = logit(np.clip(y_scaled_01, self.epsilon, 1 - self.epsilon))
        elif self.pred_transform == 'none':
            print("Skipping prediction transform...")
        else:
            raise ValueError(f"Invalid pred_transform: '{self.pred_transform}'. Choose 'log', 'logit', or 'none'.")

        # --- 2. Scaling ---
        print("Scaling data...")
        self.obs_scaler_ = StandardScaler().fit(X_full)
        self.pred_scaler_ = StandardScaler().fit(y_full)
        X_full_scaled = self.obs_scaler_.transform(X_full)
        y_full_scaled = self.pred_scaler_.transform(y_full)
        
        # --- 3. PCA ---
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
            print("Input PCA: Skipped (variance threshold >= 1.0).")
            
        if self.pred_pca_variance < 1.0:
            self.pca_pred_ = PCA(n_components=self.pred_pca_variance).fit(y_full_scaled)
            y_full_pca = self.pca_pred_.transform(y_full_scaled)
            n_pcs_pred = self.pca_pred_.n_components_
            print(f"Output PCA: Reduced {y_full.shape[1]} dims to {n_pcs_pred} components.")
        else:
            self.pca_pred_ = None
            y_full_pca = y_full_scaled
            n_pcs_pred = y_full.shape[1]
            print("Output PCA: Skipped (variance threshold >= 1.0).")

        # --- 4. Build Surrogate Components (SVD on Joint Covariance) ---
        combined_final = np.hstack((X_full_pca, y_full_pca))
        print("Building surrogate components (Covariance + SVD)...")
        
        if np.isnan(combined_final).any() or np.isinf(combined_final).any():
            raise ValueError("Input to surrogate contains NaN or Inf values after pre-processing.")
            
        mean_combined = np.mean(combined_final, axis=0)
        mean_o_final = mean_combined[:n_pcs_obs]
        mean_s_final = mean_combined[n_pcs_obs:]
        joint_cov = np.cov(combined_final.T)
        joint_cov_reg = joint_cov + np.identity(joint_cov.shape[0]) * self.regularization
        
        try:
            U, s_vals, Vt = np.linalg.svd(joint_cov_reg)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"SVD failed even after regularization: {e}")
            
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
        print("DSI Surrogate fitting complete.")
        return self

    # --- Save and Load Methods ---
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
        status = "fitted" if instance.is_fitted_ else "NOT fitted"
        print(f"Surrogate loaded from '{filename}' ({status}).")
        return instance

    # --- Diagnostic & Bias Correction Methods ---
    def diagnose_surrogate_bias(self, obs_prior, pred_prior, indices_to_plot=None, n_samples=None, figsize=(10, 8)):
        """
        Runs the surrogate on the prior observations to compare surrogate predictions 
        against the true (physics-based) prior predictions.
        """
        if not self.is_fitted_:
            raise RuntimeError("Surrogate model must be fitted before running diagnostics.")

        print("--- Running Surrogate Bias Diagnostics ---")

        if n_samples is not None and n_samples < obs_prior.shape[0]:
            idx = np.random.choice(obs_prior.shape[0], n_samples, replace=False)
            obs_subset = obs_prior[idx]
            pred_subset_true = pred_prior[idx]
        else:
            obs_subset = obs_prior
            pred_subset_true = pred_prior

        if self.log_transform_obs:
            h_obs_proc = np.log10(np.maximum(obs_subset, self.epsilon))
        else:
            h_obs_proc = obs_subset
            
        h_obs_scaled = self.obs_scaler_.transform(h_obs_proc)
        
        if self.pca_obs_:
            h_target = self.pca_obs_.transform(h_obs_scaled)
        else:
            h_target = h_obs_scaled

        M_obs = self.surrogate_components_['M_obs']
        M_pred = self.surrogate_components_['M_pred']
        mean_o = self.surrogate_components_['mean_o']
        mean_s = self.surrogate_components_['mean_s']
        n_comp = self.surrogate_components_['n_components']
        
        reg_factor = 1e-6 
        lhs = M_obs.T @ M_obs + reg_factor * np.identity(n_comp)
        rhs = (h_target - mean_o) @ M_obs 
        X_est_T = solve(lhs, rhs.T, assume_a='pos') 
        X_est = X_est_T.T 

        pred_s_pca = mean_s + (X_est @ M_pred.T)
        
        if self.pca_pred_:
            pred_s_scaled = self.pca_pred_.inverse_transform(pred_s_pca)
        else:
            pred_s_scaled = pred_s_pca
            
        pred_s_trans = self.pred_scaler_.inverse_transform(pred_s_scaled)
        
        if self.pred_transform == 'log':
            pred_subset_surrogate = 10**pred_s_trans
        elif self.pred_transform == 'logit':
            y_01 = expit(pred_s_trans)
            a, b = self.pred_bounds
            pred_subset_surrogate = y_01 * (b - a) + a
        else:
            pred_subset_surrogate = pred_s_trans

        if indices_to_plot is None:
            indices_to_plot = [0, 1, 2, 3] if pred_subset_true.shape[1] >= 4 else list(range(pred_subset_true.shape[1]))
            
        n_plots = len(indices_to_plot)
        rows = int(np.ceil(n_plots / 2))
        cols = 2 if n_plots > 1 else 1
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        
        for i, var_idx in enumerate(indices_to_plot):
            if var_idx >= pred_subset_true.shape[1]:
                continue
            ax = axes[i]
            y_true = pred_subset_true[:, var_idx]
            y_surr = pred_subset_surrogate[:, var_idx]
            corr = np.corrcoef(y_true, y_surr)[0, 1]
            min_val = min(y_true.min(), y_surr.min())
            max_val = max(y_true.max(), y_surr.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, alpha=0.7, label='1:1 Perfect')
            ax.scatter(y_true, y_surr, alpha=0.6, c='blue', edgecolor='k', s=20)
            ax.set_title(f'Variable Index {var_idx}\nCorr: {corr:.4f}')
            ax.set_xlabel('True Value (Physics)')
            ax.set_ylabel('Predicted Value (Surrogate)')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.show()
        print("Diagnostics complete.")

    def apply_bias_correction(self, posterior_ensemble, obs_prior, pred_prior, method='polynomial', poly_order=2):
        """
        Corrects the posterior predictions by learning the non-linear mapping 
        between Surrogate and Physics observed in the Prior.
        """
        if not self.is_fitted_:
            raise RuntimeError("Surrogate model must be fitted.")

        print(f"--- Applying Bias Correction ({method}) ---")
        
        # Re-run Surrogate on Prior to establish baseline
        if self.log_transform_obs:
            h_obs_proc = np.log10(np.maximum(obs_prior, self.epsilon))
        else:
            h_obs_proc = obs_prior
        
        h_obs_scaled = self.obs_scaler_.transform(h_obs_proc)
        
        if self.pca_obs_:
            h_target = self.pca_obs_.transform(h_obs_scaled)
        else:
            h_target = h_obs_scaled

        M_obs = self.surrogate_components_['M_obs']
        M_pred = self.surrogate_components_['M_pred']
        mean_o = self.surrogate_components_['mean_o']
        mean_s = self.surrogate_components_['mean_s']
        n_comp = self.surrogate_components_['n_components']
        
        reg_factor = 1e-6 
        lhs = M_obs.T @ M_obs + reg_factor * np.identity(n_comp)
        rhs = (h_target - mean_o) @ M_obs 
        X_est_T = solve(lhs, rhs.T, assume_a='pos') 
        X_est = X_est_T.T 

        pred_s_pca = mean_s + (X_est @ M_pred.T)
        
        if self.pca_pred_:
            pred_s_scaled = self.pca_pred_.inverse_transform(pred_s_pca)
        else:
            pred_s_scaled = pred_s_pca
            
        pred_s_trans = self.pred_scaler_.inverse_transform(pred_s_scaled)
        
        if self.pred_transform == 'log':
            prior_surr = 10**pred_s_trans
        elif self.pred_transform == 'logit':
            y_01 = expit(pred_s_trans)
            a, b = self.pred_bounds
            prior_surr = y_01 * (b - a) + a
        else:
            prior_surr = pred_s_trans

        corrected_posterior = np.zeros_like(posterior_ensemble)
        n_vars = pred_prior.shape[1]
        
        if posterior_ensemble.shape[1] != n_vars:
            raise ValueError("Posterior ensemble columns do not match prior predictions columns.")

        for i in range(n_vars):
            y_true = pred_prior[:, i]       
            y_surr = prior_surr[:, i]       
            y_post = posterior_ensemble[:, i] 
            
            if method == 'polynomial':
                try:
                    p = Polynomial.fit(y_surr, y_true, deg=poly_order)
                    corrected_posterior[:, i] = p(y_post)
                except:
                    corrected_posterior[:, i] = y_post 

            elif method == 'error_inflation':
                error = y_true - y_surr
                std_error = np.std(error)
                noise = np.random.normal(0, std_error, size=y_post.shape)
                corrected_posterior[:, i] = y_post + noise
        
        print("Bias correction complete.")
        return corrected_posterior

    # --- MAP/RML Solver Functions ---
    def _solve_map_analytical(self, h_target_scaled, noise_variance, x_prior_sample=None):
        n_comp = self.surrogate_components_['n_components']
        if x_prior_sample is None:
            x_prior_sample = np.zeros(n_comp)

        # Regularization: (x - x_prior)^T (x - x_prior)
        lhs = self.surrogate_components_['M_obs'].T @ self.surrogate_components_['M_obs'] + \
              noise_variance * np.identity(n_comp)
        
        rhs = (self.surrogate_components_['M_obs'].T @ \
              (h_target_scaled - self.surrogate_components_['mean_o'])) + \
              (noise_variance * x_prior_sample)
        try:
            x_map = solve(lhs, rhs, assume_a='pos')
        except np.linalg.LinAlgError:
             print("Warning: Analytical solve failed. Returning prior.")
             x_map = x_prior_sample
        return x_map

    def _solve_map_cg(self, h_target_scaled, noise_variance, x_prior_sample=None):
        n_comp = self.surrogate_components_['n_components']
        if x_prior_sample is None:
            x_prior_sample = np.zeros(n_comp)

        inv_noise_var = 1.0 / noise_variance
        M_obs_cg = self.surrogate_components_['M_obs']
        mu_o_cg = self.surrogate_components_['mean_o']
        
        def objective_function(x):
            o_pred_scaled = mu_o_cg + M_obs_cg @ x
            prior_term = np.sum((x - x_prior_sample)**2)
            likelihood_term = inv_noise_var * np.sum((h_target_scaled - o_pred_scaled)**2)
            return 0.5 * (likelihood_term + prior_term)
            
        def objective_gradient(x):
            o_pred_scaled = mu_o_cg + M_obs_cg @ x
            grad = -inv_noise_var * M_obs_cg.T @ (h_target_scaled - o_pred_scaled) + (x - x_prior_sample)
            return grad
            
        x0 = x_prior_sample
        result = minimize(fun=objective_function, x0=x0, method='CG', jac=objective_gradient,
                          options={'maxiter': 5000, 'gtol': 1e-6})
        if not result.success: print(f"Warning: CG did not converge: {result.message}")
        return result.x

    def _solve_map_ls(self, h_target_scaled, noise_variance, x_prior_sample=None):
        n_comp = self.surrogate_components_['n_components']
        if x_prior_sample is None:
            x_prior_sample = np.zeros(n_comp)

        noise_std = np.sqrt(noise_variance)
        M_obs_ls = self.surrogate_components_['M_obs']
        mu_o_ls = self.surrogate_components_['mean_o']
        
        def residual_function(x):
            data_residuals = (h_target_scaled - (mu_o_ls + M_obs_ls @ x)) / noise_std
            prior_residuals = x - x_prior_sample
            return np.concatenate((data_residuals, prior_residuals))
            
        x0 = x_prior_sample
        result = least_squares(fun=residual_function, x0=x0, method='lm', jac='2-point')
        if not result.success: print(f"Warning: least_squares did not converge: {result.message}")
        return result.x

    # --- Ensemble Solvers ---
    def _solve_ensemble_smoother(self, h_target_final, obs_noise_var_processed, n_posterior_samples):
        print(f"\n--- Performing Ensemble Smoother (ES) with {n_posterior_samples} members ---")
        M_obs, mean_o = self.surrogate_components_['M_obs'], self.surrogate_components_['mean_o']
        n_components_svd = self.surrogate_components_['n_components']
        n_pcs_obs = M_obs.shape[0]

        X_prior = np.random.normal(0.0, 1.0, size=(n_components_svd, n_posterior_samples))
        
        obs_noise_std_final = np.sqrt(np.mean(obs_noise_var_processed))
        H_true = np.tile(h_target_final.reshape(-1, 1), (1, n_posterior_samples))
        Noise_obs = np.random.normal(0.0, obs_noise_std_final, size=(n_pcs_obs, n_posterior_samples))
        H_noisy = H_true + Noise_obs
        C_noise = np.identity(n_pcs_obs) * (obs_noise_std_final**2)

        t_start = time.time()
        O_pred = (M_obs @ X_prior) + mean_o.reshape(-1, 1)
        t_end = time.time()
        print(f"  Surrogate run ({n_posterior_samples} realizations): {t_end - t_start:.4f} sec")

        X_prime = X_prior - X_prior.mean(axis=1, keepdims=True)
        O_prime = O_pred - O_pred.mean(axis=1, keepdims=True)
        
        C_oo = (O_prime @ O_prime.T) / (n_posterior_samples - 1)
        C_xo = (X_prime @ O_prime.T) / (n_posterior_samples - 1)

        matrix_to_invert = C_oo + C_noise
        try:
            K_T = solve(matrix_to_invert, C_xo.T, assume_a='pos')
            kalman_gain = K_T.T
        except np.linalg.LinAlgError:
            print("Warning: ES covariance matrix singular. Using pseudo-inverse.")
            kalman_gain = C_xo @ np.linalg.pinv(matrix_to_invert)
        
        X_posterior = X_prior + kalman_gain @ (H_noisy - O_pred)
        return X_posterior.T

    def _solve_ies(self, h_target_final, obs_noise_var_processed, n_posterior_samples, n_ies_iterations):
        """
        Solves the Iterative Ensemble Smoother using Multiple Data Assimilation (ES-MDA).
        """
        print(f"\n--- Performing ES-MDA with {n_posterior_samples} members ---")
        
        M_obs, mean_o = self.surrogate_components_['M_obs'], self.surrogate_components_['mean_o']
        n_components_svd = self.surrogate_components_['n_components']
        n_pcs_obs = M_obs.shape[0]

        X_k = np.random.normal(0.0, 1.0, size=(n_components_svd, n_posterior_samples))
        
        # --- MDA WEIGHTING FACTOR (Alpha) ---
        alpha = n_ies_iterations 
        
        obs_noise_std_final = np.sqrt(np.mean(obs_noise_var_processed))
        C_noise_inflated = np.identity(n_pcs_obs) * (obs_noise_std_final**2) * alpha
        H_true = np.tile(h_target_final.reshape(-1, 1), (1, n_posterior_samples))

        prev_phi_mean = None
        cumulative_runs = 0
        ies_start_time = time.time()
        
        print(f"{'Iter':<5} {'Mean Phi':<15} {'Std Phi':<15} {'% Reduction':<15} {'Cumul Time (s)':<15} {'Cumul Runs':<15}")
        print("-" * 85)

        for i_iter in range(n_ies_iterations):
            O_k = (M_obs @ X_k) + mean_o.reshape(-1, 1)
            
            cumulative_runs += n_posterior_samples
            current_elapsed = time.time() - ies_start_time
            
            residuals = (H_true - O_k) / obs_noise_std_final
            phi_ensemble = np.sum(residuals**2, axis=0)
            
            phi_mean = np.mean(phi_ensemble)
            phi_std = np.std(phi_ensemble)
            
            if prev_phi_mean is None:
                red_str = "  -"
            else:
                reduction = (prev_phi_mean - phi_mean) / prev_phi_mean * 100
                red_str = f"{reduction:.2f}%"

            print(f"{i_iter + 1:<5} {phi_mean:<15.4f} {phi_std:<15.4f} {red_str:<15} {current_elapsed:<15.4f} {cumulative_runs:<15}")
            
            prev_phi_mean = phi_mean

            # Resample noise every iteration (ES-MDA Requirement)
            Noise_obs = np.random.normal(0.0, obs_noise_std_final * np.sqrt(alpha), 
                                         size=(n_pcs_obs, n_posterior_samples))
            H_noisy = H_true + Noise_obs

            X_k_prime = X_k - X_k.mean(axis=1, keepdims=True)
            O_k_prime = O_k - O_k.mean(axis=1, keepdims=True)
            
            C_oo_k = (O_k_prime @ O_k_prime.T) / (n_posterior_samples - 1)
            C_xo_k = (X_k_prime @ O_k_prime.T) / (n_posterior_samples - 1)
            
            matrix_to_invert = C_oo_k + C_noise_inflated
            
            try:
                K_T = solve(matrix_to_invert, C_xo_k.T, assume_a='pos')
                kalman_gain_k = K_T.T
            except np.linalg.LinAlgError:
                kalman_gain_k = C_xo_k @ np.linalg.pinv(matrix_to_invert)

            X_k = X_k + kalman_gain_k @ (H_noisy - O_k)
        
        return X_k.T

    def _calculate_calibration_metrics(self, x_posterior_mean, h_observed_orig,
                                     avg_noise_variance_final, h_target_final):
        metrics = {'chi2_red': np.nan, 'mean_rel_error_perc': np.nan}
        n_observations_final = len(h_target_final)
        o_predicted_final = self.surrogate_components_['mean_o'] + \
                            self.surrogate_components_['M_obs'] @ x_posterior_mean
        if avg_noise_variance_final > 0:
            residuals = h_target_final - o_predicted_final
            total_chi2 = np.sum(residuals**2 / avg_noise_variance_final)
            metrics['chi2_red'] = total_chi2 / n_observations_final
        
        if self.pca_obs_:
             o_pred_scaled = self.pca_obs_.inverse_transform(o_predicted_final.reshape(1,-1))
        else:
             o_pred_scaled = o_predicted_final.reshape(1,-1)
        o_pred_transformed = self.obs_scaler_.inverse_transform(o_pred_scaled)

        if self.log_transform_obs:
            o_pred_orig = 10**o_pred_transformed
        else:
            o_pred_orig = o_pred_transformed

        abs_h_observed = np.abs(h_observed_orig.flatten())
        safe_denominator = np.where(abs_h_observed < self.epsilon, 1.0, abs_h_observed)
        rel_error = np.abs(o_pred_orig.flatten() - h_observed_orig.flatten()) / safe_denominator
        metrics['mean_rel_error_perc'] = np.mean(rel_error) * 100
        return metrics

    def predict(self, h_observed, obs_noise_std,
                inversion_type='rml', solver='analytical',
                n_posterior_samples=500, n_ies_iterations=3, 
                gd_learning_rate=1e-7, return_ensemble=False):
        """
        Performs prediction using the fitted DSI surrogate.
        """
        if not self.is_fitted_:
            raise RuntimeError("Surrogate model has not been fitted. Call .fit() first.")

        h_obs_single_orig = h_observed.copy().reshape(1, -1)
        obs_noise_std_orig = obs_noise_std.copy().reshape(1, -1)

        # --- Transform observed data and noise ---
        if self.log_transform_obs:
            h_obs_processed = np.log10(np.maximum(h_obs_single_orig, self.epsilon))
            obs_noise_std_processed = obs_noise_std_orig / (np.maximum(h_obs_single_orig, self.epsilon) * np.log(10))
            obs_noise_var_processed = obs_noise_std_processed**2
        else:
            h_obs_processed = h_obs_single_orig
            obs_noise_std_processed = obs_noise_std_orig
            obs_noise_var_processed = obs_noise_std_processed**2

        h_obs_scaled = self.obs_scaler_.transform(h_obs_processed)

        if self.pca_obs_:
             h_target_final = self.pca_obs_.transform(h_obs_scaled).flatten()
             avg_noise_variance_final = np.mean(obs_noise_var_processed)
        else:
             h_target_final = h_obs_scaled.flatten()
             avg_noise_variance_final = np.mean(obs_noise_var_processed)

        # --- Choose Solver Function ---
        solver_map = {
            'analytical': self._solve_map_analytical,
            'cg': self._solve_map_cg,
            'ls': self._solve_map_ls
        }
        if solver not in solver_map:
            raise ValueError(f"Invalid solver type '{solver}'. Choose 'analytical', 'cg', or 'ls'.")
        solve_map_func = solver_map[solver]

        # --- Perform Inversion ---
        x_posterior = None
        if inversion_type == 'map':
            print(f"\n--- Performing Single MAP Estimation using '{solver}' solver ---")
            x_posterior = solve_map_func(h_target_final, avg_noise_variance_final, None).reshape(1, -1)
            n_posterior_samples = 1

        elif inversion_type == 'rml':
            print(f"\n--- Performing RML ({n_posterior_samples} samples) using '{solver}' solver ---")
            
            if self.log_transform_obs:
                noise_gen = np.random.normal(0.0, obs_noise_std_processed, size=(n_posterior_samples, h_obs_single_orig.shape[1]))
                noisy_h_processed = h_obs_processed + noise_gen
            else:
                 noise_gen = np.random.normal(0.0, obs_noise_std_processed, size=(n_posterior_samples, h_obs_single_orig.shape[1]))
                 noisy_h_processed = h_obs_processed + noise_gen

            noisy_h_scaled = self.obs_scaler_.transform(noisy_h_processed)
            if self.pca_obs_: noisy_h_targets = self.pca_obs_.transform(noisy_h_scaled)
            else: noisy_h_targets = noisy_h_scaled

            # Generate Prior Samples for Regularization
            n_comp = self.surrogate_components_['n_components']
            x_prior_samples = np.random.normal(0.0, 1.0, size=(n_posterior_samples, n_comp))

            # --- Optimization: Improved Dask Client Management ---
            try:
                try:
                    client = get_client()
                    user_managed_client = True
                except ValueError:
                    client = Client(LocalCluster())
                    user_managed_client = False
                
                print(f"Dask dashboard at: {client.dashboard_link}")
                
                map_solver_delayed = delayed(solve_map_func)
                delayed_x_maps = [map_solver_delayed(noisy_h_targets[k, :], 
                                                     avg_noise_variance_final, 
                                                     x_prior_samples[k, :])
                                  for k in range(n_posterior_samples)]
                print(f"Executing {n_posterior_samples} MAP solves in parallel...")
                x_posterior_list = dask.compute(*delayed_x_maps)
                
                if not user_managed_client:
                    client.close()
                    
                x_posterior = np.array(x_posterior_list)
                
            except Exception as e:
                print(f"Dask parallel execution failed: {e}. Falling back to serial.")
                x_posterior_list = []
                for k in range(n_posterior_samples):
                     if (k+1)%100 == 0: print(f"  Processing RML sample {k+1}/{n_posterior_samples} (serial)...")
                     x_posterior_list.append(solve_map_func(noisy_h_targets[k, :], 
                                                            avg_noise_variance_final,
                                                            x_prior_samples[k, :]))
                x_posterior = np.array(x_posterior_list)
            print(f"RML complete. Shape of posterior x samples: {x_posterior.shape}")

        elif inversion_type == 'es':
            x_posterior = self._solve_ensemble_smoother(h_target_final, obs_noise_var_processed, n_posterior_samples)
            print(f"ES complete. Shape of posterior x samples: {x_posterior.shape}")
        
        elif inversion_type == 'ies':
            x_posterior = self._solve_ies(h_target_final, obs_noise_var_processed, n_posterior_samples, n_ies_iterations)
            print(f"IES complete. Shape of posterior x samples: {x_posterior.shape}")

        else:
            raise ValueError("Invalid inversion_type. Choose 'map', 'rml', 'es', or 'ies'.")

        if x_posterior is None or x_posterior.shape[0] == 0:
             raise RuntimeError("Posterior parameter estimation failed.")

        # --- Calculate Metrics & Generate Predictions ---
        x_posterior_mean = np.mean(x_posterior, axis=0)
        calibration_metrics = self._calculate_calibration_metrics(
            x_posterior_mean, h_obs_single_orig, avg_noise_variance_final, h_target_final
        )
        print("\n--- Calibration Performance ---")
        print(f"  Reduced Chi-squared (Chi²/N_obs_final): {calibration_metrics['chi2_red']:.4f}")
        print(f"  Mean Abs Rel Error (Original Space): {calibration_metrics['mean_rel_error_perc']:.2f}%")

        print("\n--- Generating Posterior Predictions ---")
        posterior_s_pca_scaled = self.surrogate_components_['mean_s'] + \
                                 (self.surrogate_components_['M_pred'] @ x_posterior.T).T
        if self.pca_pred_:
            posterior_s_scaled = self.pca_pred_.inverse_transform(posterior_s_pca_scaled)
        else:
            posterior_s_scaled = posterior_s_pca_scaled
        
        posterior_s_transformed = self.pred_scaler_.inverse_transform(posterior_s_scaled)

        if self.pred_transform == 'log':
            posterior_s_original = 10**posterior_s_transformed
        elif self.pred_transform == 'logit':
            y_pred_01 = expit(posterior_s_transformed)
            a, b = self.pred_bounds
            posterior_s_original = y_pred_01 * (b - a) + a
        else:
            posterior_s_original = posterior_s_transformed

        final_mean = np.mean(posterior_s_original, axis=0)
        if n_posterior_samples > 1 or x_posterior.shape[0] > 1:
            final_std = np.std(posterior_s_original, axis=0)
        else:
            final_std = None
        print("Prediction complete.")

        if return_ensemble:
            return final_mean, final_std, calibration_metrics, posterior_s_original
        else:
            return final_mean, final_std, calibration_metrics
