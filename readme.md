# 🔬 Morph Approximation Lab

Interactive Streamlit application for building and analyzing multi-factor copula models with MorphZ transformations.

## Features

- **Flexible Copula Construction**: Build multi-factor copula models with various marginal distributions
- **Tunable Correlations**: Control correlation strength between variables or use random structures
- **Multiple Distributions**: Support for Normal, Gamma, Beta, Lognormal, Exponential, Uniform, and Log-uniform
- **MorphZ Integration**: Compute morphing transformations at different orders (2-6)
- **Comprehensive Visualization**: 
  - Correlation matrices and covariance heatmaps
  - Corner plots for distribution comparison
  - Log PDF distributions
- **KL Divergence Analysis**: Evaluate approximation quality with confidence intervals

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/morph-approximation-lab.git
cd morph-approximation-lab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Add Factors**: Use the sidebar to configure and add factors with:
   - Distribution type
   - Dimension (number of variables)
   - Correlation strength (specified or random)
   - Distribution parameters for each variable

2. **Generate Samples**: Set the number of samples and run the simulation

3. **Analyze Results**: Explore multiple tabs:
   - Correlations
   - Log PDF distributions
   - Covariance matrices
   - MorphZ morphing analysis
   - KL divergence comparison

4. **MorphZ Analysis**: 
   - Select morph orders to compute
   - Choose number of samples per morph
   - Toggle directory cleanup option
   - View corner plots and KL divergence metrics

## Requirements

- Python 3.8+
- streamlit
- numpy
- scipy
- pandas
- seaborn
- matplotlib
- corner
- morphZ

## Output

The application generates a `{n}_d` folder containing:
- Transformation parameter files (`params_{order}-order_TC.json`)
- Morphed distribution samples
- KL divergence statistics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

Built with ❤️ using [Streamlit](https://streamlit.io/) and powered by [MorphZ](https://github.com/your-morphz-link)
