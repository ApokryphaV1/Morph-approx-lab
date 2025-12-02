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

    def generate_random_problem(self, target_dim,max_block):
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
            max_block = min(remaining_dim, max_block)

            block_dim = np.random.randint(1, max_block + 1) if max_block > 1 else 1
            
            # Pick a distribution
            dist_name = np.random.choice(available_dists)
            
            # Pick correlation strength (randomly either None or a float)
            use_corr = np.random.choice([True, False])
            corr_strength = np.random.uniform(0.3, 0.9) if use_corr else None
            
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

class ProductOfNormalsReference:
    """
    Reference distribution: Product of independent Gaussians fitted to each marginal.
    """
    def __init__(self, data):
        """
        Fit independent normal distributions to each dimension of data.
        
        Args:
            data: Array of shape (n_samples, n_dims)
        """
        self.means = np.mean(data, axis=0)
        self.stds = np.std(data, axis=0)
        self.n_dims = data.shape[1]
        
    def sample(self, n_samples):
        """Generate samples from the product of normals."""
        samples = np.random.normal(
            loc=self.means, 
            scale=self.stds, 
            size=(n_samples, self.n_dims)
        )
        return samples
    
    def logpdf(self, x):
        """
        Compute log PDF of the product of normals.
        
        Args:
            x: Array of shape (n_samples, n_dims)
        
        Returns:
            Array of log PDF values
        """
        x = np.atleast_2d(x)
        logpdf_vals = np.zeros(x.shape[0])
        
        for i in range(self.n_dims):
            # Log PDF of normal distribution
            logpdf_vals += stats.norm.logpdf(x[:, i], loc=self.means[i], scale=self.stds[i])
        
        return logpdf_vals

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
    max_block = st.sidebar.number_input("Max order of correlation", min_value=2, max_value=7, value=4, step=1)

    
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
            logs = st.session_state.sim.generate_random_problem(target_dims,max_block=max_block)
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
    st.markdown("Use 100-300 samples for fast results.")
    n_samples = st.number_input("Number of Samples", min_value=10, max_value=10000, value=200, step=100)
    
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
    
    
    # Tabs list
    tabs = st.tabs([
        "📊 Correlations", 
        "📈 Log PDF", 
        "🎯 MorphZ Analysis", 
        "🔥 Correlation vs MI", 
        "📉 KL Divergence",
        "✅ Acceptance Ratio"
    ])

    data = st.session_state.data
    logpdf_values = st.session_state.logpdf_values
    param_names = st.session_state.sim.get_param_names()
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
        
        # Controls for histogram
        col_hist1, col_hist2 = st.columns(2)
        with col_hist1:
            hist_bins = st.slider("Number of Bins", min_value=10, max_value=100, value=20, step=10)
        with col_hist2:
            hist_sample_limit = st.slider("Max Samples to Plot", min_value=1, max_value=len(data), value=n_samples, step=10)
        
        # Ensure finite values for plotting range
        finite_logpdf = logpdf_values[np.isfinite(logpdf_values)]
        if len(finite_logpdf) == 0:
            st.warning("All Log PDF values are non-finite (outside distribution support). Check configuration.")
        else:
            # Subsample for plotting if needed
            plot_logpdf = finite_logpdf[:hist_sample_limit] if len(finite_logpdf) > hist_sample_limit else finite_logpdf
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(plot_logpdf, bins=hist_bins, color='royalblue', edgecolor='grey')
            ax.set_title(f"Histogram of Log PDF Values ({len(plot_logpdf)} samples)")
            ax.set_xlabel("Log(PDF)")
            ax.set_ylabel("density")
            st.pyplot(fig)
            plt.close()
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Mean Log PDF", f"{np.mean(finite_logpdf):.4f}")
            with col_m2:
                st.metric("Std Log PDF", f"{np.std(finite_logpdf):.4f}")
    
# Tab 3: MorphZ Analysis (full replacement) - unified colors for corner & histogram
    with tabs[2]:

        if 'morph_data' not in st.session_state:
            st.info("No factors added yet. Use the sidebar to configure.")
        else:

            st.subheader("MorphZ Morphing Analysis")

            # Controls
            col_a, col_b = st.columns(2)
            with col_a:
                morph_orders = st.multiselect(
                    "Select Morph Orders",
                    [1, 2, 3, 4, 5, 6,7,8,9,10],
                    default=[1, 2, 3],
                    key="ui_morph_orders"
                )
            with col_b:
                n_morph_samples = st.number_input(
                    "Samples per Morph", min_value=10, max_value=20000,
                    value=1000, step=100, key="ui_n_morph_samples"
                )

            remove_existing = st.checkbox(
                "🗑️ Remove existing output directory before analysis",
                value=True,
                help="If enabled, deletes the output folder before running to ensure clean results."
            )

            run_btn = st.button("Run MorphZ Analysis", type="primary")

            if run_btn:
                # basic checks
                if len(morph_orders) == 0:
                    st.warning("Please select at least one morph order.")
                elif data is None:
                    st.error("No data available. Run sampling first.")
                else:
                    with st.spinner("Computing Morph approximation transformations..."):
                        output_dir = f"{len(param_names)}_d"

                        # clean output dir optionally
                        if remove_existing and os.path.exists(output_dir):
                            try:
                                shutil.rmtree(output_dir)
                                st.info(f"🗑️ Removed existing directory: {output_dir}")
                            except Exception as e:
                                st.warning(f"Could not remove existing directory: {e}")

                        os.makedirs(output_dir, exist_ok=True)

                        # containers to fill
                        morph_data = {}
                        kl_results = {}
                        morph_logpdf_on_original = {}

                        # iterate selected orders
                        for order in morph_orders:
                            try:
                                n_dim = len(param_names)
                                try:
                                    n_blocks = math.comb(n_dim, order)
                                except Exception:
                                    n_blocks = None
                                st.info(f"⏳ Order {order}: computing total correlation (blocks = {n_blocks})")

                                # run mz.evidence (best-effort)
                                try:
                                    if order == 1:
                                        _ = mz.evidence(
                                            data,
                                            logpdf_values,
                                            st.session_state.sim.logpdf,
                                            n_resamples=2,
                                            morph_type="indep",
                                            kde_bw="isj",
                                            param_names=param_names,
                                            output_path=output_dir
                                        )
                                    else:
                                        _ = mz.evidence(
                                            data,
                                            logpdf_values,
                                            st.session_state.sim.logpdf,
                                            n_resamples=2,
                                            morph_type=f"{order}_group",
                                            kde_bw="silverman",
                                            param_names=param_names,
                                            output_path=output_dir
                                        )
                                except Exception as e_evidence:
                                    print(f"mz.evidence warning for order {order}: {e_evidence}")

                                # instantiate morph KDE object
                                try:
                                    if order == 1:
                                        morph_kde = mz.Morph_Indep(data, kde_bw="silverman")
                                    else:
                                        morph_kde = mz.GroupKDE(
                                            data,
                                            f"{output_dir}/params_{order}-order_TC.json",
                                            param_names=param_names
                                        )
                                except Exception as e_inst:
                                    st.error(f"Could not initialize morph object for order {order}: {e_inst}")
                                    continue

                                # resample from morph
                                try:
                                    m_samples = morph_kde.resample(n_morph_samples)
                                except Exception as e_res:
                                    st.error(f"Resample failed for order {order}: {e_res}")
                                    continue

                                # compute morph logpdf on morph samples
                                try:
                                    if order == 1:
                                        morph_logpdf_morph = morph_kde.logpdf_kde(m_samples.T)
                                    else:
                                        morph_logpdf_morph = morph_kde.logpdf(m_samples.T)
                                except Exception as e_logm:
                                    st.warning(f"Could not compute morph logpdf for order {order}: {e_logm}")
                                    morph_logpdf_morph = np.full(m_samples.shape[0], -1e10)

                                # evaluate original logpdf at morph samples (for acceptance table analyses)
                                try:
                                    original_logpdf_at_morph_samples = st.session_state.sim.logpdf(m_samples)
                                    morph_logpdf_on_original[order] = np.atleast_1d(original_logpdf_at_morph_samples).astype(float)
                                except Exception as e_eval:
                                    st.warning(f"Could not evaluate original logpdf at morph samples for order {order}: {e_eval}")
                                    morph_logpdf_on_original[order] = np.full(m_samples.shape[0], -1e10)

                                # store morph data
                                morph_data[order] = {
                                    "samples": m_samples,
                                    "logpdf": np.atleast_1d(morph_logpdf_morph).astype(float)
                                }

                                # compute morph logpdf on original data (for KL)
                                try:
                                    if order == 1:
                                        morph_logpdf_on_data = morph_kde.logpdf_kde(data.T)
                                    else:
                                        morph_logpdf_on_data = morph_kde.logpdf(data.T)
                                    morph_logpdf_on_data = np.atleast_1d(morph_logpdf_on_data).astype(float)
                                except Exception as e_m_on_d:
                                    st.warning(f"Could not compute morph logpdf on original data for order {order}: {e_m_on_d}")
                                    morph_logpdf_on_data = np.full(data.shape[0], -1e10)

                                # compute KL estimate (Original || Morph)
                                try:
                                    kl_div, kl_std = compute_kl_divergence(logpdf_values, morph_logpdf_on_data)
                                except Exception as e_kl:
                                    st.warning(f"KL computation failed for order {order}: {e_kl}")
                                    kl_div, kl_std = np.nan, np.nan

                                kl_results[order] = {
                                    "kl": float(kl_div),
                                    "std": float(kl_std),
                                    "lower_ci": float(kl_div - 1.96 * kl_std) if np.isfinite(kl_div) and np.isfinite(kl_std) else np.nan,
                                    "upper_ci": float(kl_div + 1.96 * kl_std) if np.isfinite(kl_div) and np.isfinite(kl_std) else np.nan
                                }

                                st.success(f"Completed Morph order {order}")
                            except Exception as e_outer:
                                st.error(f"Unhandled error for order {order}: {e_outer}")
                                continue

                        # --- Reference: Product of independent normals baseline ---
                        try:
                            ref_dist = ProductOfNormalsReference(data)
                            ref_logpdf_on_data = ref_dist.logpdf(data)
                            kl_ref, kl_ref_std = compute_kl_divergence(logpdf_values, ref_logpdf_on_data)

                            kl_results["ref"] = {
                                "kl": float(kl_ref),
                                "std": float(kl_ref_std),
                                "lower_ci": float(kl_ref - 1.96 * kl_ref_std),
                                "upper_ci": float(kl_ref + 1.96 * kl_ref_std)
                            }

                            # store reference samples/logpdf for plotting
                            ref_samples = ref_dist.sample(n_morph_samples)
                            morph_data["ref"] = {
                                "samples": ref_samples,
                                "logpdf": ref_dist.logpdf(ref_samples)
                            }
                            st.success("Computed Reference baseline (Product of Normals).")
                        except Exception as e_ref:
                            st.warning(f"Reference baseline failed: {e_ref}")

                        # save to session state for other tabs/plots
                        st.session_state.morph_data = morph_data
                        st.session_state.kl_results = kl_results
                        st.session_state.morph_logpdf_on_original = morph_logpdf_on_original

            # After run (or if previously run) — visualization & analysis
            st.markdown("---")
            st.header("Morph Results & Visualizations")

            # guard if no morphs computed yet
            available_morphs = list(st.session_state.get("morph_data", {}).keys())
            if len(available_morphs) == 0:
                st.info("No morphs computed yet. Use 'Run MorphZ Analysis' to compute morph approximations (and the Reference baseline).")
            else:
                # Unified, robust shared toggles that modify session state
                st.subheader("Visualization toggles (shared)")
                ucol1, ucol2 = st.columns(2)

                default_selected = available_morphs.copy()
                with ucol1:
                    plot_dims = st.slider("Number of plotted parameters:", min_value=2, max_value=st.session_state.sim.total_dim, value=min(st.session_state.sim.total_dim,8), step=1)
                    show_original = True
                    show_morphs = True
                    # create the multiselect with fixed key so we can change via buttons
                    st.multiselect(
                        "Select morphs to display (shared)",
                        options=available_morphs,
                        default=default_selected,
                        key="sel_shared"
                    )

                selected_morphs_shared = st.session_state.get("sel_shared", default_selected)
                # Ensure selection contains only currently available keys
                selected_morphs_shared = [k for k in selected_morphs_shared if k in available_morphs]

                # Define a single color palette to be used for both corner & hist plots
                palette_colors = ['#1f77b4',  # blue (original)
                                '#d62728',  # red
                                '#2ca02c',  # green
                                '#ff7f0e',  # orange
                                '#9467bd',  # purple
                                '#8c564b',  # brown
                                '#e377c2',  # pink
                                '#17becf']  # cyan

                # Map each displayed key (excluding original) to a color deterministically
                morph_color_map = {}
                for idx, key in enumerate(selected_morphs_shared):
                    # +1 to reserve palette_colors[0] for original
                    morph_color_map[key] = palette_colors[(idx + 1) % len(palette_colors)]

                # --- Corner plot (uses same toggles & unified contour levels) ---
                st.subheader("Corner Plot Comparison")
                try:
                    # choose dims to plot (safe guard)
                    data_plot = data[:, :plot_dims]
                    labels_plot = param_names[:plot_dims]

                    # Define unified contour levels (these are credible fractions used by corner)
                    # Pick whichever set suits your needs; these are typical choices (50%, 75%, 95%)
                    contour_levels = [0.5, 0.75, 0.95]

                    fig = None
                    if show_original:
                        # Use palette_colors[0] for original
                        fig = corner.corner(
                            data_plot,
                            labels=labels_plot,
                            color=palette_colors[0],
                            hist_kwargs={"density": True},
                            show_titles=True,
                            title_fmt=".2f",
                            levels=contour_levels,
                            fill_contours=True,
                            plot_contours=True,
                            smooth=1.0,
                            contour_kwargs={"linewidths": 1.2}
                        )

                    if show_morphs and len(selected_morphs_shared) > 0:
                        for idx, key in enumerate(selected_morphs_shared):
                            # if user selected the original key accidentally (rare), skip - original is plotted separately above
                            if key == "original" or key == "data":
                                continue
                            mdata = st.session_state.morph_data[key]
                            m_samples_plot = mdata['samples'][:, :plot_dims]
                            color = morph_color_map.get(key, palette_colors[(idx + 1) % len(palette_colors)])
                            corner.corner(
                                m_samples_plot,
                                labels=labels_plot,
                                fig=fig,
                                color=color,
                                hist_kwargs={"density": True},
                                show_titles=True,
                                title_fmt=".2f",
                                levels=contour_levels,
                                fill_contours=True,
                                plot_contours=True,
                                smooth=1.0,
                                contour_kwargs={"linewidths": 1.2}
                            )

                    # legend
                    legend_lines = []
                    legend_labels = []
                    if show_original:
                        legend_lines.append(plt.Line2D([0], [0], color=palette_colors[0], lw=2))
                        legend_labels.append("Original")
                    if show_morphs:
                        for key in selected_morphs_shared:
                            color = morph_color_map.get(key)
                            legend_lines.append(plt.Line2D([0], [0], color=color, lw=2))
                            legend_labels.append("Reference" if key == "ref" else f"Morph {key}")

                    if fig is not None:
                        fig.legend(legend_lines, legend_labels, loc="upper right", fontsize=12)
                        st.pyplot(fig)
                        plt.close()
                except Exception as e_corner:
                    st.error(f"Error generating corner plot: {e_corner}")

                # --- Log-PDF histogram (uses same toggles & colors) ---
                st.subheader("Log PDF Comparison")
                try:
                    finite_logpdf_values = logpdf_values[np.isfinite(logpdf_values)]
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if show_original and finite_logpdf_values.size > 0:
                        ax.hist(finite_logpdf_values - finite_logpdf_values.mean(),
                                density=True, bins=30, label="Original", alpha=0.9, color=palette_colors[0])

                    if show_morphs and len(selected_morphs_shared) > 0:
                        # fixed the typo here and iterate properly
                        for idx, key in enumerate(selected_morphs_shared):
                            mdata = st.session_state.morph_data[key]
                            morph_logpdf = mdata["logpdf"]
                            finite_morph_logpdf = morph_logpdf[np.isfinite(morph_logpdf)]
                            if finite_morph_logpdf.size == 0:
                                continue
                            label = "Reference" if key == "ref" else f"Morph {key}"
                            color = morph_color_map.get(key, palette_colors[(idx + 1) % len(palette_colors)])
                            ax.hist(finite_morph_logpdf - finite_morph_logpdf.mean(),
                                    density=True, bins=30, alpha=0.5,
                                    label=label, color=color)

                    ax.set_xlabel("Log PDF (mean subtracted)")
                    ax.set_ylabel("Density")
                    ax.set_title("Log PDF Comparison")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e_hist:
                    st.error(f"Error generating log-PDF comparison: {e_hist}")



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
            # Toggle for showing/hiding reference in plots
            col_toggle1, col_toggle2 = st.columns([3, 1])
            with col_toggle1:
                st.markdown("### KL Divergence: D<sub>KL</sub>(Original || Approximation)", unsafe_allow_html=True)
            with col_toggle2:
                show_reference_in_plot = st.checkbox("Show Reference in Plot", value=True, key="show_ref_plot")
            
            # Retain the original note about the source of negative values
            st.markdown("*Lower values indicate better approximation. **Negative values shown are raw Monte Carlo estimates; the true KL divergence is non-negative.** The KL value is clipped to zero for ranking and plotting.*")
            
            # Create DataFrame for table with Reference first
            kl_data = []
            
            # Add reference first if it exists
            if 'ref' in st.session_state.kl_results:
                res = st.session_state.kl_results['ref']
                kl_data.append({
                    "Approximation": "Ref (Prod. Normals\n(sample means and stds))",
                    "KL Divergence (Clipped)": max(0, res['kl']),
                    "KL Divergence (Raw)": res['kl'],
                    "Std Error": res['std'],
                    "95% CI Lower": res['lower_ci'],
                    "95% CI Upper": res['upper_ci']
                })
            
            # Then add morph orders
            for order in sorted([o for o in st.session_state.kl_results.keys() if o != 'ref']):
                res = st.session_state.kl_results[order]
                kl_data.append({
                    "Approximation": f"Morph Order {order}",
                    "KL Divergence (Clipped)": max(0, res['kl']),
                    "KL Divergence (Raw)": res['kl'],
                    "Std Error": res['std'],
                    "95% CI Lower": res['lower_ci'],
                    "95% CI Upper": res['upper_ci']
                })
            
            kl_df = pd.DataFrame(kl_data)
            
            # Display table
            st.dataframe(
                kl_df.style.format({
                    "KL Divergence (Clipped)": "{:.6f}",
                    "KL Divergence (Raw)": "{:.6f}",
                    "Std Error": "{:.6f}",
                    "95% CI Lower": "{:.6f}",
                    "95% CI Upper": "{:.6f}"
                }).highlight_min(subset=["KL Divergence (Clipped)"], color='lightgreen'),
                use_container_width=True
            )
            
            # Plot KL divergence with error bars
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Build plot items based on toggle
            plot_items = []
            plot_labels = []
            
            # Add reference first if it exists AND toggle is on
            if 'ref' in st.session_state.kl_results and show_reference_in_plot:
                plot_items.append('ref')
                plot_labels.append('Reference')
            
            # Then add numeric orders in sorted order
            for order in sorted([o for o in st.session_state.kl_results.keys() if o != 'ref']):
                plot_items.append(order)
                plot_labels.append(f"Order {order}")
            
            # Use the clipped KL values for the bar height
            kl_values_raw = [st.session_state.kl_results[o]['kl'] for o in plot_items]
            kl_values_clipped = [max(0, kl) for kl in kl_values_raw]
            kl_stds = [st.session_state.kl_results[o]['std'] for o in plot_items]
            
            x_pos = np.arange(len(plot_items))
            # Color reference bar differently
            bar_colors = ['gray' if o == 'ref' else 'olive' for o in plot_items]
            
            # Plot using the clipped values
            bars = ax.bar(x_pos, kl_values_clipped, color=bar_colors, alpha=0.7, edgecolor='black', yerr=kl_stds, capsize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_labels, rotation=45, ha='right')
            ax.set_ylabel("KL Divergence (Clipped at 0)", fontsize=12)
            ax.set_xlabel("Approximation", fontsize=12)
            ax.set_title("KL Divergence: Original || Approximation (Raw Estimates clipped to 0)", fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            # Add a zero line
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero line')
            
            # Add value labels on bars (using the original raw values for display)
            for i, (bar, val_raw, val_clipped, std) in enumerate(zip(bars, kl_values_raw, kl_values_clipped, kl_stds)):
                # If clipped value is > 0, place label at clipped height + std, else at 0 - std (to show the negative estimate)
                height = bar.get_height() 
                y_pos = height + std if val_raw >= 0 else 0 - std 
                
                # Show the raw estimate in the label
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{val_raw:.4f}',
                        ha='center', va='bottom' if val_raw >= 0 else 'top', 
                        fontsize=10, fontweight='bold')
            
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Summary statistics
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            # Summary always includes ALL computed results (toggle doesn't affect this)
            summary_items = st.session_state.kl_results
            
            if len(summary_items) == 0:
                st.warning("No approximations to summarize.")
            else:
                col1, col2, col3 = st.columns(3)
                
                # Find the best/worst order based on the CLIPPED KL value
                kl_results_clipped = {order: max(0, res['kl']) for order, res in summary_items.items()}
                
                with col1:
                    # Find best based on clipped KL
                    best_order = min(summary_items.keys(), key=lambda x: kl_results_clipped[x])
                    # Display the CLIPPED KL value
                    best_kl_clipped = kl_results_clipped[best_order]
                    best_std = summary_items[best_order]['std']
                    best_label = "Reference" if best_order == 'ref' else f"Order {best_order}"
                    st.metric("Best Approximation", best_label, 
                              f"KL = {best_kl_clipped:.6f} ± {best_std:.6f}")
                with col2:
                    # Find worst based on clipped KL
                    worst_order = max(summary_items.keys(), key=lambda x: kl_results_clipped[x])
                    # Display the CLIPPED KL value
                    worst_kl_clipped = kl_results_clipped[worst_order]
                    worst_std = summary_items[worst_order]['std']
                    worst_label = "Reference" if worst_order == 'ref' else f"Order {worst_order}"
                    st.metric("Worst Approximation", worst_label,
                              f"KL = {worst_kl_clipped:.6f} ± {worst_std:.6f}")
                with col3:
                    # Calculate average based on CLIPPED KL values
                    avg_kl_clipped = np.mean(list(kl_results_clipped.values()))
                    st.metric("Average KL (Clipped)", f"{avg_kl_clipped:.6f}")
                
                # Additional insight: Show all approximations ranked
                st.markdown("---")
                st.subheader("Ranking by KL Divergence (Lower is Better)")
                
                ranking_data = []
                for order in sorted(summary_items.keys(), key=lambda x: kl_results_clipped[x]):
                    res = summary_items[order]
                    kl_clipped = kl_results_clipped[order]
                    label = "Reference (Prod. Normals)" if order == 'ref' else f"Morph Order {order}"
                    ranking_data.append({
                        "Rank": len(ranking_data) + 1,
                        "Approximation": label,
                        "KL Divergence (Clipped)": kl_clipped,
                        "Std Error": res['std']
                    })
                
                ranking_df = pd.DataFrame(ranking_data)
                st.dataframe(
                    ranking_df.style.format({
                        "KL Divergence (Clipped)": "{:.6f}",
                        "Std Error": "{:.6f}"
                    }).background_gradient(subset=["KL Divergence (Clipped)"], cmap='RdYlGn_r'),
                    use_container_width=True
                )

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
