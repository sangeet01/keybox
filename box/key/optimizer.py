"""
KeyBox Optimizer: Formulation Optimization Engine (v0.8.0)
Box-Behnken Design, Simplex-Centroid Mixture, and Combined Optimization.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .models import EnvironmentalConditions, DosageFormConfig, DosageForm, MoleculeType
from .core_engine import KeyBoxSystem
from .designer import create_enhanced_molecule_from_smiles, ENHANCED_EXCIPIENT_LIBRARY

try:
    from pyDOE3 import bbdesign
    HAS_PYDOE = True
except ImportError:
    HAS_PYDOE = False


def generate_box_behnken(n_factors: int, center_points: int = 3) -> np.ndarray:
    """Box-Behnken design generator. Falls back to built-in for 3 factors."""
    if HAS_PYDOE:
        return bbdesign(n_factors, center=center_points)
    if n_factors == 3:
        design = np.array([
            [-1, -1,  0], [ 1, -1,  0], [-1,  1,  0], [ 1,  1,  0],
            [-1,  0, -1], [ 1,  0, -1], [-1,  0,  1], [ 1,  0,  1],
            [ 0, -1, -1], [ 0,  1, -1], [ 0, -1,  1], [ 0,  1,  1],
        ], dtype=float)
        centers = np.zeros((center_points, 3))
        return np.vstack([design, centers])
    raise NotImplementedError("Built-in BBD supports 3 factors. Install pyDOE3 for more.")


class FormulationOptimizer:
    """Ratio and excipient screening optimizer."""

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def optimize_excipient_ratio(self, api_smiles: str, excipient_smiles: str,
                                 ratio_range: Tuple[float, float] = (0.1, 10.0),
                                 env_conditions: Optional[EnvironmentalConditions] = None,
                                 n_points: int = 20) -> Dict:
        """Find optimal API:Excipient ratio by maximizing OCS."""
        env = env_conditions or EnvironmentalConditions()
        ratios = np.logspace(np.log10(ratio_range[0]), np.log10(ratio_range[1]), n_points)
        ocs_scores = []

        print(f"[OPTIMIZER] Testing {n_points} ratios...")
        for ratio in ratios:
            system = KeyBoxSystem(voxel_bounds=(20.0, 20.0, 20.0), resolution=1.0,
                                  env_conditions=env, random_state=self.random_state)
            api = create_enhanced_molecule_from_smiles(api_smiles, "API")
            exc = create_enhanced_molecule_from_smiles(excipient_smiles, "Excipient")
            api.concentration = ratio / (1.0 + ratio)
            exc.concentration = 1.0 / (1.0 + ratio)
            system.add_molecule(api, is_api=True)
            system.add_molecule(exc, is_api=False)
            results = system.analyze_enhanced_formulation(include_uncertainty=False)
            ocs_scores.append(results['metrics']['OCS'])

        best_idx = np.argmax(ocs_scores)
        print(f"[OPTIMIZER] Optimal ratio: {ratios[best_idx]:.2f} (OCS={ocs_scores[best_idx]:.2f})")
        return {
            'optimal_ratio': ratios[best_idx],
            'best_ocs': ocs_scores[best_idx],
            'ratios': ratios,
            'ocs_curve': np.array(ocs_scores),
            'classification': 'Compatible' if ocs_scores[best_idx] >= 60 else 'Incompatible'
        }

    def screen_excipient_combinations(self, api_smiles: str,
                                      excipient_library: Optional[Dict] = None,
                                      n_excipients: int = 2,
                                      top_k: int = 5,
                                      env_conditions: Optional[EnvironmentalConditions] = None) -> List[Dict]:
        """Screen multi-excipient combinations ranked by OCS."""
        library = excipient_library or ENHANCED_EXCIPIENT_LIBRARY
        env = env_conditions or EnvironmentalConditions()
        from itertools import combinations
        combos = list(combinations(library.keys(), n_excipients))
        print(f"[OPTIMIZER] Screening {len(combos)} combinations...")
        results = []
        for combo in combos:
            system = KeyBoxSystem(voxel_bounds=(20.0, 20.0, 20.0), resolution=1.0,
                                  env_conditions=env, random_state=self.random_state)
            api = create_enhanced_molecule_from_smiles(api_smiles, "API")
            api.concentration = 0.5
            system.add_molecule(api, is_api=True)
            exc_conc = 0.5 / n_excipients
            for exc_name in combo:
                exc_data = library[exc_name]
                exc_smiles = exc_data[0] if isinstance(exc_data, tuple) else exc_data
                exc = create_enhanced_molecule_from_smiles(exc_smiles, exc_name)
                exc.concentration = exc_conc
                system.add_molecule(exc, is_api=False)
            analysis = system.analyze_enhanced_formulation(include_uncertainty=False)
            results.append({
                'excipients': combo,
                'ocs': analysis['metrics']['OCS'],
                'classification': analysis['classification'],
                'mechanisms': analysis['mechanisms']
            })
        results.sort(key=lambda x: x['ocs'], reverse=True)
        print(f"[OPTIMIZER] Top: {results[0]['excipients']} (OCS={results[0]['ocs']:.2f})")
        return results[:top_k]


class KeyBoxOptimizer:
    """
    Advanced multi-factorial formulation optimizer (v0.8.0).
    Supports BBD (process factors) and Simplex-Centroid (mixture components)
    in a unified Combined Optimization run.
    """

    def __init__(self, api_smiles: str, api_name: str = "API",
                 api_type: MoleculeType = MoleculeType.SMALL_MOLECULE):
        self.api_smiles = api_smiles
        self.api_name = api_name
        self.api_type = api_type
        self.factor_names = []
        self.factor_ranges = []
        self.mixture_components = []
        self.fixed_excipients = {}

    def add_factor(self, name: str, low: float, high: float):
        """Add an independent process factor (e.g., Temperature, Moisture)."""
        self.factor_names.append(name)
        self.factor_ranges.append((low, high))

    def add_mixture_component(self, name: str, smiles: str):
        """Add a mixture component. All mixture components must sum to 1.0."""
        self.mixture_components.append((name, smiles))

    def add_fixed_excipient(self, name: str, smiles: str):
        """Add a fixed excipient at standard level."""
        self.fixed_excipients[name] = smiles

    def _decode_design_point(self, coded_point: np.ndarray) -> Dict:
        """Convert coded coordinates to real formulation parameters."""
        n_process = len(self.factor_names)
        n_mixture = len(self.mixture_components)
        real_params = {}

        if n_process > 0:
            process_part = coded_point[:n_process]
            for i, name in enumerate(self.factor_names):
                low, high = self.factor_ranges[i]
                real_params[name] = low + (high - low) * (process_part[i] + 1) / 2

        if n_mixture > 0:
            mixture_part = coded_point[n_process:].copy()
            if mixture_part.sum() > 0:
                mixture_part = mixture_part / mixture_part.sum()
            else:
                mixture_part = np.zeros(n_mixture)
                mixture_part[0] = 1.0
            for i, (name, _) in enumerate(self.mixture_components):
                real_params[name] = mixture_part[i]

        return real_params

    def _run_keybox_simulation(self, real_params: Dict, resolution: float = 1.5,
                               n_bootstrap: int = 15) -> Dict:
        """
        Run one KeyBox simulation with BOOTSTRAP SMOOTHING
        to eliminate voxel discretization noise.
        """
        system = KeyBoxSystem(voxel_bounds=(20.0, 20.0, 20.0),
                              resolution=resolution, random_state=42)
        api = create_enhanced_molecule_from_smiles(self.api_smiles, self.api_name, self.api_type)
        system.add_molecule(api, is_api=True)

        for name, smiles in self.mixture_components:
            exc = create_enhanced_molecule_from_smiles(smiles, name)
            exc.concentration = real_params.get(name, 0.0)
            system.add_molecule(exc, is_api=False)

        for name, smiles in self.fixed_excipients.items():
            exc = create_enhanced_molecule_from_smiles(smiles, name)
            system.add_molecule(exc, is_api=False)

        system.env_conditions = EnvironmentalConditions(
            pH=real_params.get("pH", 7.0),
            moisture=real_params.get("Moisture", 0.5),
            temperature=real_params.get("Temperature", 25.0)
        )

        results = system.analyze_enhanced_formulation(include_uncertainty=True, n_mc=n_bootstrap)
        ocs_mean = results['confidence_intervals'].get('mean', results['metrics']['OCS'])
        return {"ocs": ocs_mean, "classification": results.get("classification", "Unknown"),
                "mechanisms": results.get("mechanisms", []), "full_results": results}

    def _generate_simplex_centroid(self, n_components: int) -> np.ndarray:
        """Generate Simplex-Centroid design: vertices, binary blends, centroid."""
        from itertools import combinations
        points = []
        for i in range(n_components):
            p = np.zeros(n_components); p[i] = 1.0
            points.append(p)
        if n_components >= 2:
            for comb in combinations(range(n_components), 2):
                p = np.zeros(n_components); p[list(comb)] = 0.5
                points.append(p)
        points.append(np.full(n_components, 1.0 / n_components))
        return np.unique(np.array(points), axis=0)

    def _generate_simplex_lattice(self, n_components: int, degree: int = 2) -> np.ndarray:
        """Generate Simplex-Lattice design for higher-degree mixture models."""
        from itertools import combinations
        points = []
        for i in range(n_components):
            p = np.zeros(n_components); p[i] = 1.0
            points.append(p)
        if degree >= 2:
            for comb in combinations(range(n_components), 2):
                p = np.zeros(n_components); p[list(comb)] = 0.5
                points.append(p)
        points.append(np.full(n_components, 1.0 / n_components))
        return np.unique(np.array(points), axis=0)

    def run_combined_optimization(self, n_center: int = 3, resolution: float = 1.5,
                                  verbose: bool = True) -> Dict:
        """
        [v0.8.0] Combined Mixture-Process optimization.
        - Simplex-Centroid design for mixture components (sum = 1.0)
        - Box-Behnken design for process factors (independent)
        - Scheffe Quadratic model + L-BFGS-B global optimizer
        """
        n_process = len(self.factor_names)
        n_mixture = len(self.mixture_components)

        if n_mixture == 0:
            return self.run_bbd_optimization(n_center, resolution, verbose)

        if n_process >= 3:
            process_design = generate_box_behnken(n_process, center_points=n_center)
        elif n_process == 2:
            process_design = np.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.],[0.,0.]])
        elif n_process == 1:
            process_design = np.array([[-1.],[1.],[0.]])
        else:
            process_design = np.array([[]])

        mixture_design = self._generate_simplex_centroid(n_mixture)

        from itertools import product as iproduct
        combined_points = []
        for p, m in iproduct(process_design, mixture_design):
            combined_points.append(np.concatenate([p, m]))
        combined_points = np.array(combined_points)

        if verbose:
            print(f"[OPT] Combined Design: {n_process} Process + {n_mixture} Mixture components.")
            print(f"[OPT] Total Points to simulate: {len(combined_points)}")

        responses = []
        mixture_names = [n for n, _ in self.mixture_components]
        for i, point in enumerate(combined_points):
            real_params = self._decode_design_point(point)
            sim = self._run_keybox_simulation(real_params, resolution=resolution)
            responses.append(sim)
            if verbose and i % 10 == 0:
                print(f"  Point {i+1} complete...")

        import pandas as pd
        design_df = pd.DataFrame(combined_points, columns=self.factor_names + mixture_names)
        design_df["OCS"] = [r["ocs"] for r in responses]

        try:
            from sklearn.linear_model import LinearRegression

            X_process = design_df[self.factor_names].values if n_process > 0 else np.zeros((len(design_df), 0))
            X_mixture = design_df[mixture_names].values
            y = design_df["OCS"].values

            def build_features(X_p, X_m):
                feats = [X_m]
                for i in range(n_mixture):
                    for j in range(i+1, n_mixture):
                        feats.append((X_m[:, i] * X_m[:, j]).reshape(-1, 1))
                if n_process > 0:
                    feats.append(X_p)
                    for i in range(n_process):
                        feats.append((X_p[:, i]**2).reshape(-1, 1))
                    for i in range(n_process):
                        for j in range(n_mixture):
                            feats.append((X_p[:, i] * X_m[:, j]).reshape(-1, 1))
                return np.hstack(feats)

            X_feat = build_features(X_process, X_mixture)
            model = LinearRegression(fit_intercept=(n_process > 0)).fit(X_feat, y)

            def objective(x):
                x_p = x[:n_process].reshape(1, -1) if n_process > 0 else np.zeros((1, 0))
                x_m = x[n_process:].reshape(1, -1)
                if x_m.sum() > 0:
                    x_m = x_m / x_m.sum()
                return -model.predict(build_features(x_p, x_m))[0]

            bounds = [(-1.1, 1.1)] * n_process + [(0., 1.)] * n_mixture
            x0 = np.concatenate([np.zeros(n_process), np.full(n_mixture, 1.0/n_mixture)])
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x[n_process:]) - 1.0}]
            result = minimize(objective, x0=x0, bounds=bounds, constraints=cons)
            optimal_real = self._decode_design_point(result.x)

            if verbose:
                print(f"[OPT] Optimization Complete. Best Predicted OCS: {-result.fun:.2f}")
                print(f"[OPT] Optimal Formulation: {optimal_real}")

            return {
                "design_df": design_df,
                "optimal_params": optimal_real,
                "predicted_ocs": -result.fun,
                "model": model
            }
        except ImportError:
            print("[WARNING] scikit-learn not found. Returning raw design results.")
            return {"design_df": design_df}

    def run_bbd_optimization(self, n_center: int = 3, resolution: float = 1.5,
                             verbose: bool = True) -> Dict:
        """BBD-only optimization for independent process factors."""
        n_factors = len(self.factor_names)
        if n_factors < 3:
            raise ValueError("BBD requires at least 3 factors.")

        coded_design = generate_box_behnken(n_factors, center_points=n_center)
        import pandas as pd
        design_df = pd.DataFrame(coded_design, columns=self.factor_names)

        if verbose:
            print(f"[BBD] Running design with {len(coded_design)} points...")

        responses = []
        for i, row in enumerate(coded_design):
            real_params = self._decode_design_point(row)
            sim = self._run_keybox_simulation(real_params, resolution=resolution)
            responses.append(sim)
            if verbose and i % 5 == 0:
                print(f"  Point {i+1} complete...")

        design_df["OCS"] = [r["ocs"] for r in responses]

        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            X = design_df[self.factor_names].values
            y = design_df["OCS"].values
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            model = LinearRegression().fit(X_poly, y)

            def objective(x):
                return -model.predict(poly.transform(x.reshape(1, -1)))[0]

            result = minimize(objective, x0=np.zeros(n_factors),
                              bounds=[(-1.1, 1.1)] * n_factors, method='L-BFGS-B')
            optimal_real = self._decode_design_point(result.x)

            if verbose:
                print(f"[BBD] Optimization Complete. Best Predicted OCS: {-result.fun:.2f}")
                print(f"[BBD] Optimal Configuration: {optimal_real}")

            return {"design_df": design_df, "optimal_params": optimal_real,
                    "predicted_ocs": -result.fun, "model": model}
        except ImportError:
            print("[WARNING] scikit-learn not found. Returning raw design results.")
            return {"design_df": design_df}
