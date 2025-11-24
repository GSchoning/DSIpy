import warnings
# Suppress warnings aggressively
warnings.filterwarnings("ignore")

import os
import shutil
import subprocess
import numpy as np
import flopy
import gstools as gs
import matplotlib.pyplot as plt

# --- Configuration ---
N_REALIZATIONS = 1000
MODEL_DIR = 'unconfined_workspace'
GRID_SHAPE = (50, 50)
DELR = 100.0
BIN_DIR = os.path.abspath("mf6_bin")

# --- 1. Setup Binary ---
def setup_mf6():
    if not os.path.exists(BIN_DIR): os.makedirs(BIN_DIR)
    mf6_path = shutil.which("mf6") or os.path.join(BIN_DIR, "mf6")
    if not os.path.exists(mf6_path):
        try: flopy.utils.get_modflow(bindir=BIN_DIR)
        except: pass
        mf6_path = os.path.join(BIN_DIR, "mf6")
    if os.name != 'nt' and os.path.exists(mf6_path): os.chmod(mf6_path, 0o755)
    return mf6_path

# --- 2. Model Runner ---
def run_unconfined_model(realization_id, mf6_exe):
    name = f'model_{realization_id}'
    sim_ws = os.path.join(MODEL_DIR, name)
    if os.path.exists(sim_ws): shutil.rmtree(sim_ws)
    os.makedirs(sim_ws)

    sim = flopy.mf6.MFSimulation(sim_name=name, version='mf6', exe_name=mf6_exe, sim_ws=sim_ws)
    tdis = flopy.mf6.ModflowTdis(sim, time_units='DAYS', perioddata=[[1.0, 1, 1.0]])
    
    # Robust Solver
    ims = flopy.mf6.ModflowIms(sim, complexity='COMPLEX', 
                               outer_dvclose=1e-2, inner_dvclose=1e-3,
                               outer_maximum=300, inner_maximum=100,
                               linear_acceleration='BICGSTAB',
                               backtracking_number=5)
    
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True, 
                               newtonoptions="NEWTON UNDER_RELAXATION")
    
    col_idx = np.arange(GRID_SHAPE[1])
    top_elev = np.linspace(60, 40, GRID_SHAPE[1])
    top_grid = np.tile(top_elev, (GRID_SHAPE[0], 1))
    
    # Deep bottom (-50m)
    dis = flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=GRID_SHAPE[0], ncol=GRID_SHAPE[1], 
                                  delr=DELR, delc=DELR, top=top_grid, botm=-50.0)
    
    seed = 7000 + realization_id 
    x = y = range(GRID_SHAPE[0])
    model_gs = gs.Exponential(dim=2, var=1, len_scale=15.0)
    srf = gs.SRF(model_gs, seed=seed)
    field_log = srf.structured([x, y])
    
    k_field = 2.0 * 10**(field_log) 
    
    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=k_field)
    ic = flopy.mf6.ModflowGwfic(gwf, strt=top_grid) 
    rch = flopy.mf6.ModflowGwfrcha(gwf, recharge=0.0005) 
    
    chd_spd = [[(0, i, 0), 45.0] for i in range(GRID_SHAPE[0])]
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)

    # --- WELL CURTAIN (Line along River) ---
    # Generate random total rate
    rng = np.random.RandomState(seed + 999)
    total_rate = rng.uniform(-20000.0, -20.0)
    
    # 5 Wells in a vertical line at Column 12
    # Rows: 10, 17, 25, 32, 40 (Spread out)
    well_col = 12
    well_rows = [10, 17, 25, 32, 40]
    rate_per_well = total_rate / len(well_rows)
    
    wel_spd = [[(0, r, well_col), rate_per_well] for r in well_rows]
    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd)
    
    oc = flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{name}.hds", saverecord=[('HEAD', 'ALL')])
    
    sim.write_simulation(silent=True)
    
    try:
        success, _ = sim.run_simulation(silent=True)
    except:
        success = False
    
    if not success: return None, None, None

    hds = gwf.output.head().get_data(kstpkper=(0, 0))
    try: shutil.rmtree(sim_ws)
    except: pass

    # Observations (Scattered)
    obs_rows = [10, 40, 25, 5, 45, 15, 35, 20, 30, 10]
    obs_cols = [10, 40, 10, 25, 45, 35, 15, 40, 10, 25]
    observations = hds[0, obs_rows, obs_cols]

    # Receptor (Further downstream at Col 40)
    rec_row, rec_col = 25, 40
    initial_h = top_grid[rec_row, rec_col]
    final_h = hds[0, rec_row, rec_col]
    
    if final_h < -100: 
        drawdown = initial_h - (-50.0)
    else:
        drawdown = initial_h - final_h
        
    return observations, drawdown, total_rate

if __name__ == "__main__":
    mf6_exe = setup_mf6()
    print(f"Generating {N_REALIZATIONS} realizations (Line of Wells)...")
    
    obs_list = []
    pred_list = []
    rate_list = []
    
    for i in range(N_REALIZATIONS):
        if i % 10 == 0: print(f"  Running {i}/{N_REALIZATIONS}...")
        o, p, r = run_unconfined_model(i, mf6_exe)
        
        if o is not None and np.max(np.abs(o)) < 1000:
            obs_list.append(o)
            pred_list.append(p)
            rate_list.append(r)
            
    print(f"\nSuccessful Models: {len(obs_list)} / {N_REALIZATIONS}")
    
    if len(obs_list) == 0:
        raise RuntimeError("All models failed.")

    obs_prior = np.array(obs_list)
    pred_prior = np.array(pred_list).reshape(-1, 1)
    rates_prior = np.array(rate_list)
    
    # Truth Selection
    target = np.percentile(pred_prior, 85)
    true_idx = (np.abs(pred_prior - target)).argmin()
        
    field_data = obs_prior[true_idx] + np.random.normal(0, 0.2, size=obs_prior.shape[1])
    true_prediction = pred_prior[true_idx]
    true_rate = rates_prior[true_idx]
    
    np.save('obs_prior_uc.npy', obs_prior)
    np.save('pred_prior_uc.npy', pred_prior)
    np.save('field_data_uc.npy', field_data)
    np.save('true_val_uc.npy', true_prediction)
    
    print(f"Selected Truth ID: {true_idx}")
    print(f"True Pumping Rate: {true_rate:.2f} m3/d")
    print(f"True Drawdown:     {true_prediction.item():.2f} m")
    print("Done.")
