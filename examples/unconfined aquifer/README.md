# Unconfined Aquifer Test Case: Non-Linear Inversion & Bias Correction

This example demonstrates the application of **DSIpy** to a challenging, non-linear groundwater problem: estimating drawdown in a heterogeneous unconfined aquifer under heavy stress.

## 🎯 Objective

To benchmark the capability of **ES-MDA (Iterative Ensemble Smoother)** and **Quantile Mapping Bias Correction** in handling non-linear systems where standard linear inversion methods typically fail.

## 🌊 Physical Setup

The synthetic reality is a 2D unconfined aquifer simulated using **MODFLOW 6**.

* **Domain:** $50 \times 50$ grid (5 km $\times$ 5 km) with a sloping land surface (60m to 40m).
* **Hydrogeology:**
    * **Heterogeneity:** Hydraulic Conductivity ($K$) is a spatially correlated Log-Normal random field ($\sigma^2_{\ln K} = 0.5$, correlation length = 1500m).
    * **Unconfined Flow:** Transmissivity varies with saturated thickness ($T = K \times b$). As drawdown occurs, the aquifer thins, reducing $T$ and accelerating drawdown. This creates a non-linear feedback loop distinct from linear confined aquifers.
* **Boundary Conditions:**
    * **River:** Constant Head Boundary ($h=45$m) on the Western edge.
    * **Recharge:** Diffuse recharge ($0.001$ m/d).
* **Stress (The "Line of Wells"):**
    * A vertical curtain of 5 pumping wells located at Column 12 (1.2 km from the river).
    * **Total Extraction:** Randomly varying between -2000 and -6000 m³/day (~730 to 2190 ML/year) per realization.
    * This creates a deep "trough" of depression that intercepts flow to the river.

## 🔮 The Prediction

We aim to predict **Drawdown** at a sensitive receptor located downstream (Row 25, Column 40).

* **Why it's hard:** The receptor is far from the stress. The drawdown propagation depends entirely on the unknown connectivity of the heterogeneous $K$ field.
* **The Non-Linearity:** In low-$K$ realizations, the aquifer thins significantly, causing drawdown to "blow up" non-linearly. The linear surrogate will likely under-predict these extreme events or fail to capture the curvature of the response, requiring bias correction.

## 📂 Files

* `generate_ensemble.py`: Generates the Prior Ensemble.
    * Runs 100 stochastic MODFLOW 6 models using the Newton-Raphson solver for stability.
    * Extracts Observations (Heads at 10 wells) and Predictions (Receptor Drawdown).
    * Saves data to `.npy` files.
* `run_inversion.py`: Performs the DSI Analysis.
    * Trains the SVD Surrogate.
    * Inverts using ES-MDA (4 iterations).
    * Diagnoses non-linearity and applies **Quantile Mapping** correction.
    * Plots the final validation against the synthetic truth.
* `plot_setup.py`: Visualizes the model domain (Wells, River, Receptor).
* `plot_context.py`: Plots the 2D Conductivity and Head fields for a single realization to show the heterogeneity and cone of depression.

## 🚀 How to Run

1.  **Generate the Ensemble:**
    ```bash
    python generate_ensemble.py
    ```
    *(This will download the MF6 binary automatically and run for ~2-5 minutes)*.

2.  **Run the Inversion:**
    ```bash
    python run_inversion.py
    ```
    *(This will display the diagnostic plots and final posterior distribution)*.

3.  **Visualize the Physics:**
    ```bash
    python plot_context.py
    ```

## 📊 Expected Results

1.  **Diagnostic Plot:** You should see a curved "Banana" relationship between the Surrogate and True Physics, confirming the non-linearity of the unconfined flow.
2.  **Bias Correction:** The `auto` correction mode should detect this non-linearity and apply Quantile Mapping.
3.  **Validation:** The corrected posterior distribution should accurately encompass the "True" drawdown value (dashed black line), correctly characterizing the risk of extreme drawdown events.
