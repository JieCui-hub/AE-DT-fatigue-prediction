"""Internal validation, feature attribution and pre-rafted domain-shift calculations."""

from __future__ import annotations
import argparse
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPOSITORY_ROOT / 'data' / 'Supplementary_Data.xlsx'
import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
SEED = 20260716
N_REPEATS = 30
N_SPLITS = 5
REPEAT_SEEDS = [SEED + i for i in range(N_REPEATS)]
STATE_ORDER = ['Base', 'Base + AE', 'Base + MS', 'Base + AE + MS']
MODEL_ORDER = ['XGBoost', 'ExtraTrees', 'RandomForest', 'GradientBoosting']

def make_model(name, random_state):
    if name == 'XGBoost':
        return XGBRegressor(n_estimators=120, max_depth=2, learning_rate=0.035, subsample=0.9, colsample_bytree=0.9, min_child_weight=1.0, reg_alpha=0.02, reg_lambda=2.0, objective='reg:squarederror', random_state=random_state, n_jobs=1, verbosity=0)
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(n_estimators=160, max_depth=6, min_samples_leaf=1, max_features=0.9, random_state=random_state, n_jobs=1)
    if name == 'RandomForest':
        return RandomForestRegressor(n_estimators=160, max_depth=6, min_samples_leaf=1, max_features=0.9, random_state=random_state, n_jobs=1)
    return GradientBoostingRegressor(n_estimators=120, learning_rate=0.035, max_depth=2, min_samples_leaf=1, subsample=0.9, loss='huber', random_state=random_state)

def load_data(data_path):
    core = pd.read_excel(data_path, sheet_name='Source_Small_Sample_91.2%_AE').copy()
    shifted = pd.read_excel(data_path, sheet_name='Pre_raft').copy()
    if 'Rafting time (h)' not in shifted.columns and 'Unnamed: 20' in shifted.columns:
        shifted = shifted.rename(columns={'Unnamed: 20': 'Rafting time (h)'})
    core['Rafting time (h)'] = 0
    schema = {'id': 'Specimen ID', 'orientation': 'Orientation', 'shared': ['Orientation', 'Temperature (C)', 'Max Stress (MPa)', 'Stress Ratio'], 'ae': ['Mutation Point', 'D_early', 'Delta_p'], 'ms': ['gamma_prime_volume_fraction_pct', 'average_gamma_prime_edge_length_um', 'average_gamma_channel_width_um'], 'rafting': 'Rafting time (h)', 'target': 'log10_Nf'}
    required = [schema['id'], *schema['shared'], *schema['ae'], *schema['ms'], schema['rafting'], schema['target']]
    for label, frame, expected in (('source', core, 28), ('pre-rafted', shifted, 6)):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f'{label} table is missing required columns: {missing}')
        if len(frame) != expected:
            raise ValueError(f'Expected {expected} rows in the {label} table, found {len(frame)}')
    core = core.reset_index(drop=True)
    shifted = shifted.reset_index(drop=True)
    return (core, shifted, schema)

def state_features(schema):
    shared, ae, ms = (list(schema['shared']), list(schema['ae']), list(schema['ms']))
    return {'Base': shared, 'Base + AE': shared + ae, 'Base + MS': shared + ms, 'Base + AE + MS': shared + ae + ms}

def encode_matrix(frame, features, orientation):
    matrix = pd.get_dummies(frame[features], columns=[orientation], prefix='Orientation', dtype=float)
    matrix.columns = [str(column).replace('[', '').replace(']', '') for column in matrix.columns]
    conditions = [str(name) for name in features[1:4]]
    orientation_columns = ['Orientation_001', 'Orientation_011', 'Orientation_111']
    for column in orientation_columns:
        if column not in matrix:
            matrix[column] = 0.0
    return matrix[conditions + orientation_columns + [str(name) for name in features[4:]]].astype(float)

def oof_task(repeat, seed, state, model_name, matrix, target, specimen_ids):
    prediction = np.full(len(target), np.nan)
    fold_ids = np.zeros(len(target), dtype=int)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(np.arange(len(target))), start=1):
        model = make_model(model_name, seed + fold * 1009)
        model.fit(matrix[train], target[train])
        prediction[test] = model.predict(matrix[test])
        fold_ids[test] = fold
    mse = float(mean_squared_error(target, prediction))
    metric = {'Repeat': repeat, 'Seed': seed, 'State': state, 'Model': model_name, 'R2': float(r2_score(target, prediction)), 'MSE_log10': mse, 'RMSE_log10': math.sqrt(mse), 'MAE_log10': float(mean_absolute_error(target, prediction))}
    rows = [{'Repeat': repeat, 'Seed': seed, 'State': state, 'Model': model_name, 'Fold': int(fold_ids[i]), 'Sample index': i + 1, 'Specimen ID': specimen_ids[i], 'Measured log10_Nf': target[i], 'OOF predicted log10_Nf': prediction[i], 'Absolute error log10': abs(target[i] - prediction[i])} for i in range(len(target))]
    return (metric, rows)

def internal_results(core, schema):
    features = state_features(schema)
    matrices = {state: encode_matrix(core, names, schema['orientation']).to_numpy(float) for state, names in features.items()}
    target = core[schema['target']].to_numpy(float)
    specimen_ids = core[schema['id']].astype(str).to_numpy()
    jobs = [delayed(oof_task)(repeat, seed, state, model, matrices[state], target, specimen_ids) for repeat, seed in enumerate(REPEAT_SEEDS, start=1) for state in STATE_ORDER for model in MODEL_ORDER]
    blocks = Parallel(n_jobs=-1, backend='loky')(jobs)
    metrics = pd.DataFrame([block[0] for block in blocks])
    predictions = pd.DataFrame([row for block in blocks for row in block[1]])
    consensus = predictions.groupby(['State', 'Model', 'Sample index', 'Specimen ID', 'Measured log10_Nf'], as_index=False, sort=False)['OOF predicted log10_Nf'].mean().rename(columns={'OOF predicted log10_Nf': 'Consensus predicted log10_Nf'})
    summary_rows = []
    for state in STATE_ORDER:
        for model_name in MODEL_ORDER:
            block = consensus.loc[consensus['State'].eq(state) & consensus['Model'].eq(model_name)]
            repeat_block = metrics.loc[metrics['State'].eq(state) & metrics['Model'].eq(model_name)]
            actual = block['Measured log10_Nf'].to_numpy(float)
            predicted = block['Consensus predicted log10_Nf'].to_numpy(float)
            mse = float(mean_squared_error(actual, predicted))
            summary_rows.append({'State': state, 'Model': model_name, 'Consensus R2': float(r2_score(actual, predicted)), 'Consensus MSE_log10': mse, 'Consensus RMSE_log10': math.sqrt(mse), 'Consensus MAE_log10': float(mean_absolute_error(actual, predicted)), 'Median repeat R2': repeat_block['R2'].median(), 'R2 2.5%': repeat_block['R2'].quantile(0.025), 'R2 97.5%': repeat_block['R2'].quantile(0.975), 'Median repeat MSE_log10': repeat_block['MSE_log10'].median()})
    summary = pd.DataFrame(summary_rows)
    baseline = metrics.loc[metrics['State'].eq('Base'), ['Repeat', 'Seed', 'Model', 'R2', 'MSE_log10']].rename(columns={'R2': 'Base R2', 'MSE_log10': 'Base MSE_log10'})
    gains = metrics.loc[~metrics['State'].eq('Base')].merge(baseline, on=['Repeat', 'Seed', 'Model'])
    gains['Delta R2 vs Base'] = gains['R2'] - gains['Base R2']
    gains['Delta MSE vs Base'] = gains['MSE_log10'] - gains['Base MSE_log10']
    return (metrics, predictions, consensus, summary, gains)

def display_feature(name):
    mapping = {'Mutation Point': 'Mutation point', 'D_early': 'D_early', 'Delta_p': 'Delta_P_RA', 'Max Stress (MPa)': 'Maximum stress', 'average_gamma_prime_edge_length_um': 'gamma-prime edge length', 'average_gamma_channel_width_um': 'gamma-channel width'}
    return mapping.get(name, name)

def normalize_features(matrix):
    values = matrix.to_numpy(float)
    normalized = np.zeros_like(values)
    for index in range(values.shape[1]):
        low, high = np.quantile(values[:, index], [0.05, 0.95])
        normalized[:, index] = 0.5 if high <= low + 1e-12 else np.clip((values[:, index] - low) / (high - low), 0.0, 1.0)
    return normalized

def shap_table(core, schema):
    matrix = encode_matrix(core, state_features(schema)['Base + AE + MS'], schema['orientation'])
    x = matrix.to_numpy(float)
    y = core[schema['target']].to_numpy(float)
    model = make_model('ExtraTrees', SEED)
    model.fit(x, y)
    rng = np.random.default_rng(SEED + 9000)
    n_samples, n_features = x.shape
    n_permutations = 256
    permutations = np.vstack([rng.permutation(n_features) for _ in range(n_permutations)])
    background = x[rng.integers(0, n_samples, size=n_permutations)].copy()
    baseline = model.predict(background)
    rows = np.arange(n_permutations)
    shap_values = np.zeros((n_samples, n_features))
    for sample_index in range(n_samples):
        current = background.copy()
        previous = baseline.copy()
        contributions = np.zeros((n_permutations, n_features))
        for step in range(n_features):
            selected = permutations[:, step]
            current[rows, selected] = x[sample_index, selected]
            updated = model.predict(current)
            contributions[rows, selected] += updated - previous
            previous = updated
        shap_values[sample_index] = contributions.mean(axis=0)
    normalized = normalize_features(matrix)
    ae_names, ms_names = (set(schema['ae']), set(schema['ms']))
    output = []
    for i in range(n_samples):
        for j, feature in enumerate(matrix.columns):
            group = 'AE' if feature in ae_names else 'MS' if feature in ms_names else 'Base'
            output.append({'Sample index': i + 1, 'Specimen ID': str(core.iloc[i][schema['id']]), 'Feature': feature, 'Display feature': display_feature(feature), 'Feature group': group, 'Feature value': x[i, j], 'Normalized feature value': normalized[i, j], 'SHAP value': shap_values[i, j], 'Absolute SHAP value': abs(shap_values[i, j])})
    return pd.DataFrame(output)

def external_task(seed, state, model_name, train_x, test_x, train_y):
    model = make_model(model_name, seed)
    model.fit(train_x, train_y)
    return (seed, state, model_name, model.predict(test_x))

def external_results(core, shifted, schema):
    features = state_features(schema)
    matrices = {}
    for state, names in features.items():
        encoded = encode_matrix(pd.concat([core[names], shifted[names]], ignore_index=True), names, schema['orientation'])
        matrices[state] = (encoded.iloc[:len(core)].to_numpy(float), encoded.iloc[len(core):].to_numpy(float))
    train_y = core[schema['target']].to_numpy(float)
    test_y = shifted[schema['target']].to_numpy(float)
    jobs = [delayed(external_task)(seed, state, model_name, *matrices[state], train_y) for seed in REPEAT_SEEDS for state in STATE_ORDER for model_name in MODEL_ORDER]
    blocks = Parallel(n_jobs=-1, backend='loky')(jobs)
    rows = []
    for seed, state, model_name, prediction in blocks:
        for i, value in enumerate(prediction):
            rows.append({'Seed': seed, 'State': state, 'Model': model_name, 'Sample index': i + 1, 'Specimen ID': str(shifted.iloc[i][schema['id']]), 'Rafting time (h)': int(shifted.iloc[i][schema['rafting']]), 'Measured log10_Nf': test_y[i], 'Predicted log10_Nf': value, 'Absolute error log10': abs(test_y[i] - value)})
    predictions = pd.DataFrame(rows)
    model_consensus = predictions.groupby(['State', 'Model', 'Sample index', 'Specimen ID', 'Rafting time (h)', 'Measured log10_Nf'], as_index=False, sort=False)['Predicted log10_Nf'].mean().rename(columns={'Predicted log10_Nf': 'Consensus predicted log10_Nf'})
    ensemble = model_consensus.groupby(['State', 'Sample index', 'Specimen ID', 'Rafting time (h)', 'Measured log10_Nf'], as_index=False, sort=False)['Consensus predicted log10_Nf'].mean().rename(columns={'Consensus predicted log10_Nf': 'Four-model ensemble predicted log10_Nf'})
    ensemble['Absolute error log10'] = np.abs(ensemble['Measured log10_Nf'] - ensemble['Four-model ensemble predicted log10_Nf'])
    metric_rows = []
    for state in STATE_ORDER:
        for model_name in MODEL_ORDER:
            block = model_consensus.loc[model_consensus['State'].eq(state) & model_consensus['Model'].eq(model_name)]
            actual = block['Measured log10_Nf'].to_numpy(float)
            predicted = block['Consensus predicted log10_Nf'].to_numpy(float)
            metric_rows.append({'State': state, 'Model': model_name, 'n_test': len(block), 'R2': r2_score(actual, predicted), 'MAE_log10': mean_absolute_error(actual, predicted), 'RMSE_log10': math.sqrt(mean_squared_error(actual, predicted))})
        block = ensemble.loc[ensemble['State'].eq(state)]
        actual = block['Measured log10_Nf'].to_numpy(float)
        predicted = block['Four-model ensemble predicted log10_Nf'].to_numpy(float)
        metric_rows.append({'State': state, 'Model': 'Four-model ensemble', 'n_test': len(block), 'R2': r2_score(actual, predicted), 'MAE_log10': mean_absolute_error(actual, predicted), 'RMSE_log10': math.sqrt(mean_squared_error(actual, predicted))})
    return (ensemble, pd.DataFrame(metric_rows))

def calculate_panel(panel, data_path):
    core, shifted, schema = load_data(data_path)
    if panel in {'4b', '4c', '4d', '4e', '4f'}:
        metrics, predictions, consensus, summary, gains = internal_results(core, schema)
        if panel in {'4b', '4c', '4d'}:
            state = {'4b': 'Base + AE', '4c': 'Base + MS', '4d': 'Base + AE + MS'}[panel]
            return consensus.loc[consensus['State'].eq(state)].merge(summary, on=['State', 'Model'], how='left')
        return summary if panel == '4e' else gains
    if panel == '4g':
        return shap_table(core, schema)
    ensemble, metrics = external_results(core, shifted, schema)
    return metrics if panel == '4n' else ensemble

def main():
    parser = argparse.ArgumentParser(description='Export one Figure 4 calculation table.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Input workbook (default: repository data/Supplementary_Data.xlsx).')
    parser.add_argument('--panel', choices=['4b', '4c', '4d', '4e', '4f', '4g', '4l', '4m', '4n'], required=True)
    parser.add_argument('--output', type=Path, help='Output .csv or .xlsx file (default: repository results/<module>_<panel>.csv).')
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f'Input workbook not found: {args.data}')
    if args.output is None:
        args.output = REPOSITORY_ROOT / 'results' / f'{Path(__file__).stem}_{args.panel}.csv'
    if args.output.suffix.lower() not in {'.csv', '.xlsx'}:
        parser.error('--output must end in .csv or .xlsx')
    result = calculate_panel(args.panel, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == '.xlsx':
        result.to_excel(args.output, index=False)
    else:
        result.to_csv(args.output, index=False)
    print(f'{args.panel}: {len(result)} rows -> {args.output}')

if __name__ == '__main__':
    main()
