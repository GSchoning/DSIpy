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

Then, in your Python script (ensure the file is named `DSIpy.py`):
```python
from DSIpy import DSISurrogate
```

## ⚡ Quick Start

### 1. Standard Uncertainty Quantification (RML)
This example demonstrates how to train the surrogate on a prior ensemble and predict uncertainty for field data.

```python
import numpy as np
from DSIpy import DSISurrogate

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

### The BEL Philosophy
**Bayesian Evidential Learning (BEL)** represents a paradigm shift from traditional model-based inversion.
* **Traditional Inversion ($d \to m \to h$):** You iteratively adjust model parameters ($m$) to match observed data ($d$), then run the model forward to get predictions ($h$). This is often computationally expensive and ill-posed.
* **BEL / DSI ($d \to h$):** We acknowledge that the physical model is just a mechanism to generate a statistical relationship between data and predictions. DSI learns this relationship directly from a **prior ensemble** of model realizations.

### How DSI Works
Data Space Inversion (DSI) constructs a statistical surrogate model based on the **joint covariance** of the observations and predictions.

1.  **The Prior:** We generate $N$ realizations of the physical model. Each realization produces a vector of simulated observations ($d$) and a target prediction ($h$).
2.  **Dimension Reduction:** We perform Principal Component Analysis (PCA) on $d$ and $h$ separately to reduce noise and dimensionality, resulting in $d^*$ and $h^*$.
3.  **The Joint Surrogate:** We concatenate $d^*$ and $h^*$ and perform a Singular Value Decomposition (SVD). This reveals a low-dimensional **latent space** ($x$) that drives the variability in *both* the data and the prediction.

The resulting linear surrogate model is:

$$
\begin{bmatrix} d \\ h \end{bmatrix} \approx \begin{bmatrix} \mu_d \\ \mu_h \end{bmatrix} + \begin{bmatrix} M_d \\ M_h \end{bmatrix} x
$$

Where:
* $M_d, M_h$ are basis matrices derived from the prior ensemble covariance.
* $x$ is a vector of latent parameters with a prior distribution $x \sim N(0, I)$.

### The "Inversion" Step
Since the surrogate relates observations $d$ directly to the latent variables $x$ via a linear operator ($M_d$), finding the posterior becomes a standard linear-Gaussian inversion problem.

When we observe real field data ($d_{obs}$), we solve for the optimal latent vector $x_{post}$ that minimizes the mismatch. Because $x$ controls both $d$ and $h$, determining $x_{post}$ automatically determines the posterior prediction $h_{post}$.

### Why use DSI?
* **Speed:** Once the ensemble is generated, DSI inversion takes seconds, whereas traditional history matching might take weeks.
* **Uncertainty:** DSI naturally preserves the geologic variability of the prior. It doesn't collapse the solution to a single "best fit," but provides a posterior probability distribution (P10, P50, P90).
* **Non-Linearity:** By using methods like the **Iterative Ensemble Smoother (IES)**, DSI can handle mild non-linearities in the data-prediction relationship.

## License

[MIT License](LICENSE)

## Author

**G. Schoning**
*Office of Groundwater Impact Assessment/Flinders University
