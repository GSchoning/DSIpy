# DSIpy: Data Space Inversion & Bayesian Evidential Learning

**DSIpy** is a lightweight, high-performance Python module for **Data Space Inversion (DSI)** and **Bayesian Evidential Learning (BEL)**. 

It provides a framework for quantifying uncertainty in complex physical systems (hydrogeology, geophysics, reservoir engineering) by learning a direct statistical relationship between observations and predictions, bypassing the need for high-dimensional parameter inversion.

## 🚀 Key Features

* **Surrogate Modeling:** Uses SVD-based dimension reduction and covariance analysis to link prior observations to predictions.
* **Multiple Inversion Strategies:**
    * **MAP:** Maximum A Posteriori estimation (Analytical & Conjugate Gradient solvers).
    * **RML:** Randomized Maximum Likelihood for robust uncertainty quantification.
    * **ES:** Ensemble Smoother for rapid, linear-Gaussian updates.
    * **IES:** Iterative Ensemble Smoother for handling mild non-linearities.
* **Parallel Computing:** Integrated **Dask** support for parallelizing RML inversions across cores.
* **Data Transformations:** Built-in handling for Log10 (orders of magnitude) and Logit (bounded variables like saturation) transforms.
* **Model Persistence:** Save and load trained surrogate models to/from disk (Pickle format).

## 📦 Installation

### Prerequisites
DSIpy requires Python 3.8+ and the following dependencies:

```bash
pip install numpy pandas scipy scikit-learn dask[distributed] matplotlib
```

### Setup
Since DSIpy is a standalone module, you can simply clone this repository and import the class:

```bash
git clone [https://github.com/yourusername/dsipy.git](https://github.com/yourusername/dsipy.git)
cd dsipy
```

Then, in your Python script:
```python
from dsi_surrogate import DSISurrogate
```

## ⚡ Quick Start

### 1. Standard Uncertainty Quantification (RML)
This example demonstrates how to train the surrogate on a prior ensemble and predict uncertainty for field data.

```python
import numpy as np
from dsi_surrogate import DSISurrogate

# 1. Load your data (Example shapes)
# obs_prior: (500 realizations, 1000 time steps)
# pred_prior: (500 realizations, 5000 grid cells)
obs_prior = np.load('prior_observations.npy')
pred_prior = np.load('prior_predictions.npy')
field_data = np.load('field_data.npy')

# 2. Initialize the Surrogate
# We use PCA to reduce dimensions and log-transforms for wide-range data
dsi = DSISurrogate(
    obs_pca_variance=0.95,   # Keep 95% energy of inputs
    pred_pca_variance=0.99,  # Keep 99% energy of outputs
    log_transform_obs=True,  # Log10 transform input
    pred_transform='log'     # Log10 transform output
)

# 3. Fit the model
dsi.fit(obs_prior, pred_prior)

# 4. Predict (Invert)
# Run Randomized Maximum Likelihood with Dask parallelism
mean, std, metrics = dsi.predict(
    h_observed=field_data,
    obs_noise_std=field_data * 0.05,  # Assume 5% noise
    inversion_type='rml',
    solver='cg',
    n_posterior_samples=100
)

print(f"Calibration Error: {metrics['mean_rel_error_perc']:.2f}%")
```

### 2. Handling Bounded Variables (e.g., Saturation)
Use the `logit` transform and `pred_bounds` to ensure predictions respect physical limits (e.g., 0.0 to 1.0).

```python
dsi_sat = DSISurrogate(
    log_transform_obs=True,
    pred_transform='logit',
    pred_bounds=[0.0, 1.0] 
)

dsi_sat.fit(obs_prior, pred_prior)

mean, std, _ = dsi_sat.predict(
    h_observed=field_data,
    obs_noise_std=noise_vec,
    inversion_type='es' # Ensemble Smoother is fast for this
)
```

## 📖 Documentation

### `DSISurrogate` Class Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `obs_pca_variance` | `float` | `1.0` | Variance to keep in input PCA (<1.0) or skip PCA (>=1.0). |
| `pred_pca_variance` | `float` | `1.0` | Variance to keep in output PCA. |
| `log_transform_obs` | `bool` | `True` | Apply Log10 to observations before processing. |
| `pred_transform` | `str` | `'log'` | `'log'`, `'logit'`, or `'none'`. |
| `pred_bounds` | `list` | `None` | `[min, max]` bounds. Required if using `'logit'`. |

### `predict()` Method Arguments

| Argument | Options | Description |
| :--- | :--- | :--- |
| `inversion_type` | `'map'`, `'rml'`, `'es'`, `'ies'` | The algorithm used to condition the posterior. |
| `solver` | `'analytical'`, `'cg'`, `'ls'` | The numerical solver for the optimization step (MAP/RML). |
| `n_posterior_samples` | `int` | Number of posterior realizations to generate. |
| `n_ies_iterations` | `int` | Number of iterations (only for `inversion_type='ies'`). |

## 💾 Saving and Loading

Train on a high-performance cluster, save the model, and predict on a laptop.

```python
# Save
dsi.save('my_surrogate.pkl')

# Load
dsi_loaded = DSISurrogate.load('my_surrogate.pkl')
```

## 📚 Theoretical Background

**DSI** operates on the principle that if a statistical relationship (Covariance) can be established between data and prediction in a latent space (via SVD), we can condition predictions on observed data without resolving the physical forward model.

The core surrogate equation is:
$$d_{obs} \approx \mu_{obs} + M_{obs} x$$
$$h_{pred} \approx \mu_{pred} + M_{pred} x$$

Where $x$ is a latent variable vector $x \sim N(0, I)$. Inversion becomes the task of finding the optimal $x$ given $d_{obs}$.

## License

[MIT License](LICENSE)

## Author

**G. Schoning**
*Hydrogeology & Groundwater Modeling*
