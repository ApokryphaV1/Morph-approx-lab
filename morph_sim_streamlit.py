import streamlit as st
import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal, ortho_group, uniform, loguniform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import corner
import morphZ as mz
import json
import os
import shutil
import math

class TunableCopulaSimulation:
    def __init__(self, seed=None):
        if seed:
            np.random.seed(seed)
        
        self.factors = []      
        self.total_dim = 0
        self.full_corr = None
        self.marginals = []
        
        # Metadata for naming
        self.var_names = [] 
        self._type_counters = {}

    def reset(self):
        """Resets the simulation state."""
        self.factors = []      
        self.total_dim = 0
        self.full_corr = None
        self.marginals = []
        self.var_names = [] 
        self._type_counters = {}

    def _generate_structured_correlation(self, dim, strength=None):
        """
        Generates a correlation matrix.
        If strength is None: Random structure.
        If strength is float (0.0 - 0.99): Controls tightness of correlation.
        """
        if dim == 1:
            return np.array([[1.0]])
        
        # Base: Random Orthogonal Matrix (Rotation)
        Q = ortho_group.rvs(dim)
        
        if strength is None:
            # Random eigenvalues between 0.5 and 2.0
            eig_vals = np.random.uniform(0.5, 2.0, size=dim)
        else:
            # Tunable Strength Logic
            base_val = 1.0 - strength
            peak_val = 1.0 + (dim - 1) * strength
            
            # Set first eigenvalue high, rest low
            eig_vals = np.full(dim, base_val)
            eig_vals[0] = peak_val
            
            # Add tiny jitter
            eig_vals += np.random.normal(0, 0.01, size=dim)
            eig_vals = np.abs(eig_vals)

        # Construct Covariance
        D = np.diag(eig_vals)
        cov = Q @ D @ Q.T
        
        # Normalize to Correlation Matrix
        d_inv = np.diag(1.0 / np.sqrt(np.diag(cov)))
        corr = d_inv @ cov @ d_inv
        
        return corr

    def add_factor(self, dist_name, params, dim, corr_strength=None):
        """
        Args:
            dist_name (str): 'norm', 'gamma', 'beta', 'lognorm', 'expon', 'uniform', 'loguniform'
            params (list): List of dicts for parameters.
            dim (int): Dimension of this factor.
            corr_strength (float, optional): 0.0 to 0.99 or None.
        """
        if len(params) != dim:
            raise ValueError(f"Length of params must match dim")

        # Generate Correlation for this block
        corr_matrix = self._generate_structured_correlation(dim, strength=corr_strength)
        
        # Build Marginals and Names
        factor_marginals = []
        
        for p in params:
            # Track naming
            if dist_name not in self._type_counters:
                self._type_counters[dist_name] = 0
            self._type_counters[dist_name] += 1
            unique_name = f"{dist_name}_{self._type_counters[dist_name]}"
            self.var_names.append(unique_name)

            # Initialize Distribution
            if dist_name == 'norm':
                dist = stats.norm(**p)
            elif dist_name == 'gamma':
                dist = stats.gamma(**p)
            elif dist_name == 'beta':
                dist = stats.beta(**p)
            elif dist_name == 'lognorm':
                dist = stats.lognorm(**p)
            elif dist_name == 'expon':
                dist = stats.expon(**p)
            elif dist_name == 'uniform':
                dist = stats.uniform(**p)
            elif dist_name == 'loguniform':
                dist = stats.loguniform(**p)
            else:
                raise ValueError(f"Unknown dist: {dist_name}")
            factor_marginals.append(dist)

        self.factors.append({
            'corr': corr_matrix,
            'dim': dim,
            'strength': corr_strength
        })
        self.marginals.extend(factor_marginals)
        self.total_dim += dim
        self.full_corr = None
        
        return f"Added Factor [{dist_name}]: {dim} vars | Strength={corr_strength if corr_strength is not None else 'Random'}"

    def generate_random_problem(self, target_dim):
        """
        Automatically generates a random high-dimensional problem.
        It partitions the target_dim into random blocks and assigns random distributions and parameters.
        """
        self.reset()
        available_dists = ['norm', 'gamma', 'beta', 'lognorm', 'expon', 'uniform']
        
        remaining_dim = target_dim
        logs = []

        while remaining_dim > 0:
            # Determine block size (random between 1 and remaining, capped at 5 for variety)
            max_block = min(remaining_dim, 5)
            block_dim = np.random.randint(1, max_block + 1) if max_block > 1 else 1
            
            # Pick a distribution
            dist_name = np.random.choice(available_dists)
            
            # Pick correlation strength (randomly either None or a float)
            use_corr = np.random.choice([True, False])
            corr_strength = np.random.uniform(0.1, 0.9) if use_corr else None
            
            # Generate Random Params
            params = []
            for _ in range(block_dim):
                p = {}
                if dist_name == 'norm':
                    p['loc'] = np.random.uniform(-5, 5)
                    p['scale'] = np.random.uniform(0.5, 3.0)
                elif dist_name == 'gamma':
                    p['a'] = np.random.uniform(1.0, 5.0)
                    p['scale'] = np.random.uniform(0.5, 2.0)
                elif dist_name == 'beta':
                    p['a'] = np.random.uniform(1.0, 5.0)
                    p['b'] = np.random.uniform(1.0, 5.0)
                elif dist_name == 'lognorm':
                    p['s'] = np.random.uniform(0.5, 1.0) # shape
                    p['scale'] = np.exp(np.random.uniform(0, 1)) # scale
                elif dist_name == 'expon':
                    p['scale'] = np.random.uniform(0.5, 3.0)
                elif dist_name == 'uniform':
                    start = np.random.uniform(-5, 5)
                    width = np.random.uniform(1, 10)
                    p['loc'] = start
                    p['scale'] = width
                params.append(p)
            
            # Add the factor
            msg = self.add_factor(dist_name, params, block_dim, corr_strength)
            logs.append(msg)
            remaining_dim -= block_dim
            
        return logs

    def get_param_names(self):
        return self.var_names

    def build(self):
        """Stitch the block diagonals together."""
        self.full_corr = np.eye(self.total_dim)
        curr = 0
        for f in self.factors:
            d = f['dim']
            self.full_corr[curr : curr+d, curr : curr+d] = f['corr']
            curr += d
            
    def sample(self, n_samples=1):
        if self.full_corr is None: 
            self.build()
        
        # Latent Gaussian Sample
        mvn = multivariate_normal(mean=np.zeros(self.total_dim), cov=self.full_corr)
        z = mvn.rvs(size=n_samples)
        if n_samples == 1: 
            z = z.reshape(1, -1)
            
        # Convert to Uniform (CDF)
        u = stats.norm.cdf(z)
        
        # Convert to Marginals (PPF)
        x = np.zeros_like(u)
        for i, dist in enumerate(self.marginals):
            x[:, i] = dist.ppf(u[:, i])
            
        return x

    def logpdf(self, x):
        if self.full_corr is None: 
            self.build()
        x = np.atleast_2d(x)
        
        log_marginals = np.zeros(x.shape[0])
        u_vals = np.zeros_like(x)
        
        # Marginal Log Likelihoods
        for i, dist in enumerate(self.marginals):
            # Calculate the logpdf for marginals
            marginal_logpdfs = dist.logpdf(x[:, i])
            
            # --- FIX: Replace -inf and nan with a large finite negative number ---
            # This is the critical line to prevent the ValueError
            marginal_logpdfs = np.nan_to_num(marginal_logpdfs, neginf=-1e10) 
            # -------------------------------------------------------------------
            
            log_marginals += marginal_logpdfs
            u_vals[:, i] = dist.cdf(x[:, i])

        # Copula Log Likelihood
        z_vals = stats.norm.ppf(np.clip(u_vals, 1e-9, 1-1e-9))
        
        mvn = multivariate_normal(mean=np.zeros(self.total_dim), cov=self.full_corr)
        log_pdf_mvn = mvn.logpdf(z_vals)
        log_pdf_indep = np.sum(stats.norm.logpdf(z_vals), axis=1)
        
        final_logpdf = (log_pdf_mvn - log_pdf_indep) + log_marginals
        
        return final_logpdf


def compute_kl_divergence(original_logpdf, morph_logpdf):
    """
    Compute KL divergence D_KL(Original || Morph) using Monte Carlo estimation.
    """
    # Forward KL: samples from original distribution
    log_ratio = original_logpdf - morph_logpdf
    kl_forward = np.mean(log_ratio)
    kl_std = np.std(log_ratio) / np.sqrt(len(log_ratio))
    
    return kl_forward, kl_std


# Streamlit App
st.set_page_config(page_title="Morph approximation Lab", layout="wide")

st.title("🔬 Tunable Copula Simulation for Morph approximation.")
st.markdown("Interactive tool for building and analyzing multi-factor copula models")

# Initialize session state
if 'sim' not in st.session_state:
    st.session_state.sim = TunableCopulaSimulation(seed=42)
    st.session_state.factors_added = []
    st.session_state.data = None
    st.session_state.logpdf_values = None
    st.session_state.morph_data = {}
    st.session_state.kl_results = {}
    st.session_state.morph_logpdf_on_original = {} # Store LogPDFs for the new tab

# Sidebar Configuration Mode
st.sidebar.header("⚙️ Configuration")
setup_mode = st.sidebar.radio("Setup Mode", ["Manual Construction", "Random Auto-Generation"])

st.sidebar.markdown("---")

if setup_mode == "Random Auto-Generation":
    st.sidebar.subheader("🎲 Auto-Generate Problem")
    target_dims = st.sidebar.number_input("Total Dimensions", min_value=2, max_value=100, value=10, step=1)
    
    if st.sidebar.button("⚡ Generate Random Problem"):
        try:
            # Clear previous state
            st.session_state.factors_added = []
            st.session_state.data = None
            st.session_state.logpdf_values = None
            st.session_state.morph_data = {}
            st.session_state.kl_results = {}
            st.session_state.morph_logpdf_on_original = {}
            
            # Generate
            logs = st.session_state.sim.generate_random_problem(target_dims)
            st.session_state.factors_added = logs
            st.success(f"Generated {target_dims}-dimensional problem with {len(logs)} factors!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

else:
    # MANUAL MODE
    st.sidebar.subheader("🛠️ Add Factor Manually")
    
    dist_name = st.sidebar.selectbox(
        "Distribution Type",
        ['norm', 'gamma', 'beta', 'lognorm', 'expon', 'uniform', 'loguniform']
    )

    dim = st.sidebar.number_input("Dimension", min_value=1, max_value=10, value=2)

    corr_strength_type = st.sidebar.radio(
        "Correlation Strength",
        ["Specified", "Random"]
    )

    if corr_strength_type == "Specified":
        corr_strength = st.sidebar.slider("Strength", 0.0, 0.99, 0.5, 0.01)
    else:
        corr_strength = None

    # Parameter inputs based on distribution
    st.sidebar.markdown(f"**Parameters for {dim} variables**")

    params = []
    for i in range(dim):
        st.sidebar.markdown(f"**Variable {i+1}**")
        
        if dist_name == 'norm':
            loc = st.sidebar.number_input(f"loc (mean) #{i+1}", value=0.0, key=f"norm_loc_{i}")
            scale = st.sidebar.number_input(f"scale (std) #{i+1}", value=1.0, min_value=0.01, key=f"norm_scale_{i}")
            params.append({'loc': loc, 'scale': scale})
            
        elif dist_name == 'gamma':
            a = st.sidebar.number_input(f"a (shape) #{i+1}", value=2.0, min_value=0.01, key=f"gamma_a_{i}")
            params.append({'a': a})
            
        elif dist_name == 'beta':
            a = st.sidebar.number_input(f"a (alpha) #{i+1}", value=2.0, min_value=0.01, key=f"beta_a_{i}")
            b = st.sidebar.number_input(f"b (beta) #{i+1}", value=2.0, min_value=0.01, key=f"beta_b_{i}")
            params.append({'a': a, 'b': b})
            
        elif dist_name == 'lognorm':
            s = st.sidebar.number_input(f"s (shape) #{i+1}", value=1.0, min_value=0.01, key=f"lognorm_s_{i}")
            loc = st.sidebar.number_input(f"loc #{i+1}", value=0.0, key=f"lognorm_loc_{i}")
            params.append({'s': s, 'loc': loc})
            
        elif dist_name == 'expon':
            scale = st.sidebar.number_input(f"scale #{i+1}", value=1.0, min_value=0.01, key=f"expon_scale_{i}")
            params.append({'scale': scale})
            
        elif dist_name == 'uniform':
            loc = st.sidebar.number_input(f"loc (lower bound) #{i+1}", value=0.0, key=f"uniform_loc_{i}")
            scale = st.sidebar.number_input(f"scale (range) #{i+1}", value=1.0, min_value=0.01, key=f"uniform_scale_{i}")
            params.append({'loc': loc, 'scale': scale})
            
        elif dist_name == 'loguniform':
            a = st.sidebar.number_input(f"a (lower bound) #{i+1}", value=0.01, min_value=1e-10, key=f"loguniform_a_{i}")
            b = st.sidebar.number_input(f"b (upper bound) #{i+1}", value=1.0, min_value=1e-10, key=f"loguniform_b_{i}")
            if a >= b:
                st.sidebar.error(f"Variable {i+1}: a must be < b")
            params.append({'a': a, 'b': b})

    if st.sidebar.button("➕ Add Factor"):
        try:
            msg = st.session_state.sim.add_factor(dist_name, params, dim, corr_strength)
            st.session_state.factors_added.append(msg)
            st.sidebar.success(msg)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Simulation"):
    st.session_state.sim = TunableCopulaSimulation(seed=42)
    st.session_state.factors_added = []
    st.session_state.data = None
    st.session_state.logpdf_values = None
    st.session_state.morph_data = {}
    st.session_state.kl_results = {}
    st.session_state.morph_logpdf_on_original = {}
    st.sidebar.success("Simulation reset!")
    st.rerun()

# Main area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Current Configuration")
    if st.session_state.factors_added:
        st.write(f"**Total Dimensions:** {st.session_state.sim.total_dim}")
        with st.expander("See Factor Details", expanded=True):
            for msg in st.session_state.factors_added:
                st.text(msg)
    else:
        st.info("No factors added yet. Use the sidebar to configure.")

with col2:
    st.subheader("🎲 Sample & Analyze")
    
    n_samples = st.number_input("Number of Samples", min_value=10, max_value=10000, value=2000, step=100)
    
    if st.button("🚀 Run Simulation", type="primary"):
        if st.session_state.sim.total_dim == 0:
            st.error("Please add at least one factor before running.")
        else:
            with st.spinner("Generating samples..."):
                st.session_state.sim.build()
                st.session_state.data = st.session_state.sim.sample(n_samples)
                st.session_state.logpdf_values = st.session_state.sim.logpdf(st.session_state.data)
                st.session_state.morph_data = {}
                st.session_state.kl_results = {}
                st.session_state.morph_logpdf_on_original = {} # Reset
            st.success(f"Generated {n_samples} samples!")

# Display results
if st.session_state.data is not None:
    data = st.session_state.data
    logpdf_values = st.session_state.logpdf_values
    param_names = st.session_state.sim.get_param_names()
    
    # Tabs list
    tabs = st.tabs([
        "📊 Correlations", 
        "📈 Log PDF", 
        "🎯 MorphZ Analysis", 
        "🔥 Correlation vs MI", 
        "📉 KL Divergence",
        "✅ Acceptance Ratio"
    ])
    
    # Tab 1: Correlation Heatmap
    with tabs[0]:
        st.subheader("Correlation Matrix")
        df = pd.DataFrame(data, columns=param_names)
        corr_obs = df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_obs, dtype=bool))
        sns.heatmap(corr_obs, mask=mask, annot=False, cmap='coolwarm', 
                    vmin=-1, vmax=1, ax=ax) 
        ax.set_title("Recovered Correlation Matrix from Samples")
        st.pyplot(fig)
        plt.close()
    
    # Tab 2: Log PDF
    with tabs[1]:
        st.subheader("Log PDF Distribution")
        
        # Ensure finite values for plotting range
        finite_logpdf = logpdf_values[np.isfinite(logpdf_values)]
        if len(finite_logpdf) == 0:
            st.warning("All Log PDF values are non-finite (outside distribution support). Check configuration.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(finite_logpdf, bins=30, color='skyblue', edgecolor='black')
            ax.set_title("Histogram of Log PDF Values for Samples")
            ax.set_xlabel("Log PDF")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()
            
            st.metric("Mean Log PDF", f"{np.mean(finite_logpdf):.4f}")
            st.metric("Std Log PDF", f"{np.std(finite_logpdf):.4f}")
    
    # Tab 3: MorphZ Analysis
    with tabs[2]:
        st.subheader("MorphZ Morphing Analysis")
        
        col_a, col_b = st.columns(2)
        with col_a:
            morph_orders = st.multiselect(
                "Select Morph Orders",
                [2, 3, 4, 5, 6],
                default=[2, 3]
            )
        with col_b:
            n_morph_samples = st.number_input("Samples per Morph", 1000, 10000, 4000, 500)
        
        # Toggle for removing existing directory
        remove_existing = st.checkbox(
            "🗑️ Remove existing output directory before analysis",
            value=True,
            help="If enabled, deletes the output folder before running to ensure clean results."
        )
        
        if st.button("🧬 Run MorphZ Analysis"):
            if len(morph_orders) == 0:
                st.warning("Please select at least one morph order.")
            else:
                with st.spinner("Computing Morph approximation transformations..."):
                    # Create output directory (delete if exists and toggle is on)
                    output_dir = f"{len(param_names)}_d"
                    
                    if remove_existing and os.path.exists(output_dir):
                        try:
                            shutil.rmtree(output_dir)
                            st.info(f"🗑️ Removed existing directory: {output_dir}")
                        except Exception as e:
                            st.warning(f"Could not remove existing directory: {e}")
                    
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Compute TC for each order
                    morph_data = {}
                    kl_results = {}
                    morph_logpdf_on_original = {} 
                    
                    for order in morph_orders:
                        try:
                            # Show block count info
                            n_dim = len(param_names)
                            n_blocks = math.comb(n_dim, order)
                            st.info(f"⏳ Computing Total correlation for {n_blocks} blocks ({n_dim} choose {order})...")

                            # Use mz.evidence to compute and save
                            try:
                                m = mz.evidence(
                                    data, 
                                    logpdf_values,
                                    st.session_state.sim.logpdf,
                                    n_resamples=2,
                                    morph_type=f"{order}_group",
                                    kde_bw='scott',
                                    param_names=param_names,
                                    output_path=output_dir
                                )
                                st.success(f"✅ Successfully computed Morph approx. for order {order}!")
                            except Exception as e:
                                # Continue on error in evidence computation
                                print(f"⚠️ Warning during mz.evidence for order {order}: {e}")

                            
                            # Load and resample - this is the critical part
                            morph_kde = mz.GroupKDE(
                                data, 
                                f"{output_dir}/params_{order}-order_TC.json",
                                param_names=param_names
                            )
                            morph_samples = morph_kde.resample(n_morph_samples)
                            
                            # Compute log PDFs for morph samples under morph distribution
                            morph_logpdf_morph = morph_kde.logpdf(morph_samples.T)
                            
                            # --- NEW EVALUATION STEP ---
                            # Evaluate the morph samples at the ORIGINAL logpdf function
                            original_logpdf_at_morph_samples = st.session_state.sim.logpdf(morph_samples)
                            morph_logpdf_on_original[order] = original_logpdf_at_morph_samples.flatten()
                            # ---------------------------
                            
                            morph_data[order] = {
                                'samples': morph_samples,
                                'logpdf': morph_logpdf_morph
                            }
                            
                            # Compute KL divergence: D_KL(Original || Morph)
                            morph_logpdf_on_original_kl = morph_kde.logpdf(data.T)
                            
                            # Ensure both arrays are 1D
                            if isinstance(morph_logpdf_on_original_kl, np.ndarray):
                                morph_logpdf_on_original_kl = morph_logpdf_on_original_kl.flatten()
                            
                            kl_div, kl_std = compute_kl_divergence(logpdf_values, morph_logpdf_on_original_kl)
                            
                            kl_results[order] = {
                                'kl': kl_div,
                                'std': kl_std,
                                'lower_ci': kl_div - 1.96 * kl_std,
                                'upper_ci': kl_div + 1.96 * kl_std
                            }
                            
                        except Exception as e:
                            # Skip this order entirely if it fails
                            st.error(f"Error processing order {order}: {e}")
                            continue
                    
                    # Store in session state
                    st.session_state.morph_data = morph_data
                    st.session_state.kl_results = kl_results
                    st.session_state.morph_logpdf_on_original = morph_logpdf_on_original # Store new results
                
                # Show results section
                st.markdown("---")
                
                if len(st.session_state.morph_data) == 0:
                    st.error("No morphs were successfully computed. Please check the errors above.")
                else:
                    # Show success message
                    st.success(f"✅ Successfully computed {len(st.session_state.morph_data)} out of {len(morph_orders)} morphs!")
                    
                    # Corner plot comparison
                    st.subheader("Corner Plot Comparison")
                    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
                    
                    try:
                        # Only plot first 5 dimensions if total dim is huge
                        plot_dims = min(5, st.session_state.sim.total_dim)
                        plot_indices = list(range(plot_dims))
                        
                        with st.spinner("Generating corner plot..."):
                            # Filter data for plot
                            data_plot = data[:, :plot_dims]
                            labels_plot = param_names[:plot_dims]
                            
                            fig = corner.corner(
                                data_plot, 
                                labels=labels_plot, 
                                color='blue',
                                hist_kwargs={"density": True},
                                show_titles=True, 
                                title_fmt=".2f"
                            )
                            
                            for idx, (order, mdata) in enumerate(st.session_state.morph_data.items()):
                                m_samples_plot = mdata['samples'][:, :plot_dims]
                                corner.corner(
                                    m_samples_plot,
                                    labels=labels_plot,
                                    fig=fig,
                                    color=colors[idx % len(colors)],
                                    hist_kwargs={"density": True},
                                    show_titles=True,
                                    title_fmt=".2f"
                                )
                            
                            # Legend
                            fig.legend(
                                [plt.Line2D([0], [0], color='blue', lw=2)] +
                                [plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2) 
                                 for i in range(len(st.session_state.morph_data))],
                                ['Original'] + [f'Morph {o}' for o in st.session_state.morph_data.keys()],
                                loc='upper right',
                                fontsize=14
                            )
                            
                            st.pyplot(fig)
                            if st.session_state.sim.total_dim > 5:
                                st.caption("Note: Only showing first 5 dimensions for clarity.")
                            plt.close()
                    except Exception as e:
                        st.error(f"Error generating corner plot: {e}")
                    
                    # Log PDF comparison
                    st.subheader("Log PDF Comparison")
                    try:
                        # Filter out non-finite LogPDFs for plotting
                        finite_logpdf_values = logpdf_values[np.isfinite(logpdf_values)]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        ax.hist(finite_logpdf_values - np.mean(finite_logpdf_values), 
                               density=True, bins=30, color='blue', 
                               alpha=0.8, label='Original')
                        
                        for idx, (order, mdata) in enumerate(st.session_state.morph_data.items()):
                            morph_logpdf = mdata['logpdf']
                            # Filter out non-finite LogPDFs for plotting
                            finite_morph_logpdf = morph_logpdf[np.isfinite(morph_logpdf)]
                            
                            ax.hist(finite_morph_logpdf - finite_morph_logpdf.mean(),
                                   density=True, bins=30, 
                                   color=colors[idx % len(colors)],
                                   alpha=0.4, label=f'Morph {order}')
                        
                        ax.set_xlabel('Log PDF (mean subtracted)')
                        ax.set_ylabel('Density')
                        ax.set_title('Log PDF Comparison')
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error generating log PDF plot: {e}")

    # Tab 4: Correlation vs MI
    with tabs[3]:
        st.subheader("Pearson Correlation vs Mutual Information")
        
        # Check if output directory and MI plot exist
        output_dir = f"{len(param_names)}_d"
        mi_plot_path = f"{output_dir}/mi_heatmap.png"
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### Pearson Correlation (Computed)")
            # Re-plot Pearson Correlation
            df = pd.DataFrame(data, columns=param_names)
            corr_obs = df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_obs, dtype=bool))
            sns.heatmap(corr_obs, mask=mask, annot=False, cmap='coolwarm', 
                        vmin=-1, vmax=1, ax=ax)
            ax.set_title("Pearson Correlation")
            st.pyplot(fig)
            plt.close()
        
        with c2:
            st.markdown("##### Mutual Information (MorphZ)")
            if os.path.exists(mi_plot_path):
                st.image(mi_plot_path, caption=f"Mutual Information Heatmap ({mi_plot_path})", use_container_width=True)
            else:
                st.info("⚠️ Mutual Information heatmap not found.")
                st.warning("Please run the **MorphZ Analysis** in the previous tab to generate the MI plot.")
                st.code(f"Expected Path: {mi_plot_path}")
    
    # Tab 5: KL Divergence
    with tabs[4]:
        st.subheader("KL Divergence Analysis")
        
        if len(st.session_state.kl_results) == 0:
            st.info("Run MorphZ Analysis first to compute KL divergences.")
        else:
            st.markdown("### KL Divergence: D<sub>KL</sub>(Original || Morph)", unsafe_allow_html=True)
            st.markdown("*Lower values indicate better approximation. Negative values can occur due to Monte Carlo estimation with finite samples.*")
            
            # Create DataFrame for table
            kl_df = pd.DataFrame([
                {
                    "Morph Order": order, 
                    "KL Divergence": res['kl'],
                    "Std Error": res['std'],
                    "95% CI Lower": res['lower_ci'],
                    "95% CI Upper": res['upper_ci']
                }
                for order, res in sorted(st.session_state.kl_results.items())
            ])
            
            # Display table
            st.dataframe(
                kl_df.style.format({
                    "KL Divergence": "{:.6f}",
                    "Std Error": "{:.6f}",
                    "95% CI Lower": "{:.6f}",
                    "95% CI Upper": "{:.6f}"
                }).highlight_min(subset=["KL Divergence"], color='lightgreen'),
                use_container_width=True
            )
            
            # Plot KL divergence with error bars
            fig, ax = plt.subplots(figsize=(10, 6))
            
            orders = sorted(st.session_state.kl_results.keys())
            kl_values = [st.session_state.kl_results[o]['kl'] for o in orders]
            kl_stds = [st.session_state.kl_results[o]['std'] for o in orders]
            
            x_pos = np.arange(len(orders))
            bars = ax.bar(x_pos, kl_values, color='steelblue', alpha=0.7, edgecolor='black', yerr=kl_stds, capsize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"Order {o}" for o in orders])
            ax.set_ylabel("KL Divergence", fontsize=12)
            ax.set_xlabel("Morph Order", fontsize=12)
            ax.set_title("KL Divergence: Original || Morph (with 95% CI)", fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero line')
            
            # Add value labels on bars
            for i, (bar, val, std) in enumerate(zip(bars, kl_values, kl_stds)):
                height = bar.get_height()
                y_pos = height + std if height > 0 else height - std
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{val:.4f}',
                       ha='center', va='bottom' if height > 0 else 'top', 
                       fontsize=10, fontweight='bold')
            
            ax.legend()
            st.pyplot(fig)
            plt.close()
            
            # Summary statistics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                best_order = min(st.session_state.kl_results, key=lambda x: st.session_state.kl_results[x]['kl'])
                best_kl = st.session_state.kl_results[best_order]['kl']
                best_std = st.session_state.kl_results[best_order]['std']
                st.metric("Best Approximation", f"Order {best_order}", 
                         f"KL = {best_kl:.6f} ± {best_std:.6f}")
            with col2:
                worst_order = max(st.session_state.kl_results, key=lambda x: st.session_state.kl_results[x]['kl'])
                worst_kl = st.session_state.kl_results[worst_order]['kl']
                worst_std = st.session_state.kl_results[worst_order]['std']
                st.metric("Worst Approximation", f"Order {worst_order}",
                         f"KL = {worst_kl:.6f} ± {worst_std:.6f}")
            with col3:
                avg_kl = np.mean([res['kl'] for res in st.session_state.kl_results.values()])
                st.metric("Average KL", f"{avg_kl:.6f}")

    # Tab 6: Acceptance Ratio
    with tabs[5]:
        # st.subheader("✅ Acceptance Ratio: Morph Samples in Original Distribution")
        
        morph_logpdfs = st.session_state.morph_logpdf_on_original
        
        if len(morph_logpdfs) == 0:
            st.info("Run **MorphZ Analysis** first to generate and evaluate samples.")
        else:
            
            # --- Histogram Plot ---
            # st.markdown("### Histogram of Original LogPDF Evaluated at Morph Samples")
            # st.markdown("This shows how likely the **morphed samples** are under the **original (true) distribution**.")
            
            # fig, ax = plt.subplots(figsize=(10, 6))
            # colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
            orders = sorted(morph_logpdfs.keys())
            
            # Filter original LogPDF for plotting threshold reference
            finite_original_logpdf = st.session_state.logpdf_values[np.isfinite(st.session_state.logpdf_values)]
            
            for idx, order in enumerate(orders):
                logpdfs = morph_logpdfs[order]
                # Filter out non-finite LogPDFs for plotting (already mostly handled by logpdf fix)
                finite_logpdfs = logpdfs[np.isfinite(logpdfs)]
                
                # ax.hist(finite_logpdfs, bins=50, density=True, alpha=0.6, 
                #         color=colors[idx % len(colors)], label=f'Morph Order {order}')

            # # Plot original LogPDF (data from tabs[1]) for comparison
            # ax.hist(finite_logpdf, bins=50, density=True, 
            #             histtype='step', linewidth=2, color='blue', label='Original Data LogPDF')

            # ax.set_title("Original LogPDF ($\ln p(\mathbf{x})$) at Morph Sample Locations")
            # ax.set_xlabel("Original LogPDF Value")
            # ax.set_ylabel("Density")
            # ax.legend()
            # st.pyplot(fig)
            # plt.close()

            # --- Acceptance Table ---
            st.markdown("### Acceptance Table")
            finite_original_logpdf = st.session_state.logpdf_values[np.isfinite(st.session_state.logpdf_values)]

            if len(finite_original_logpdf) > 0:
                # Define acceptance threshold based on the minimum finite original LogPDF
                min_original_logpdf = np.min(finite_original_logpdf)
            else:
                # Fallback if original logpdfs were all outside support
                min_original_logpdf = -1e10
            
            acceptance_results = []
            n_proposed = n_morph_samples # From the input field above
            
            for order in orders:
                logpdfs = morph_logpdfs[order]
                # Calculate accepted samples (LogPDF is higher than the minimum observed in true samples)
                # Since logpdfs array now contains -1e10 instead of -inf, this comparison works reliably.
                n_accepted = np.sum(logpdfs >= min_original_logpdf)
                acceptance_rate = (n_accepted / n_proposed) * 100
                
                acceptance_results.append({
                    "Morph Order": order,
                    "Proposed Samples": n_proposed,
                    "Accepted Samples": n_accepted,
                    "Acceptance Rate (%)": acceptance_rate
                })

            accept_df = pd.DataFrame(acceptance_results)
            
            st.dataframe(
                accept_df.style.format({
                    "Acceptance Rate (%)": "{:.2f}%"
                }).highlight_max(subset=["Acceptance Rate (%)"], color='lightgreen'),
                use_container_width=True
            )

st.markdown("---")
st.markdown("Built using Streamlit | Powered by MorphZ")