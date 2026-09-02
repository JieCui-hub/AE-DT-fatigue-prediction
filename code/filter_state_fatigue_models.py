"""Repeated-CV fatigue-life models, filtering-state comparisons and ablations."""

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
from sklearn.model_selection import GroupShuffleSplit, KFold, LeaveOneGroupOut
from xgboost import XGBRegressor
SEED = 20260716
N_REPEATS = 30
N_SPLITS = 5
REPEAT_SEEDS = [SEED + i for i in range(N_REPEATS)]
SOURCE_SHEET = 'Source_Small_Sample_91.2%_AE'
SPECIMEN_STATE_SHEET = 'AE_State_Descriptors'
ML_STATE_ORDER = ['No AE', 'Raw AE', '84.80% removed', '91.15% removed', '97.73% removed']
MODEL_ORDER = ['XGBoost', 'ExtraTrees', 'RandomForest', 'GradientBoosting']
AE_COLUMNS = ['Mutation Point', 'D_early', 'Delta_p']
SHARED_COLUMNS = ['Orientation', 'Temperature (C)', 'Max Stress (MPa)', 'Stress Ratio']
EVENT_STATE_SHEETS = {'Raw AE': 'AE_Raw', '84.80% removed': 'AE_Filter_84.8pct', '91.15% removed': 'AE_Filter_91.2pct', '97.73% removed': 'AE_Filter_97.7pct'}
SPECIMEN_STATE_SHEETS = {'Raw AE': 'Source_Small_Sample_Raw_AE', '84.80% removed': 'Source_Small_Sample_84.8%_AE', '91.15% removed': 'Source_Small_Sample_91.2%_AE', '97.73% removed': 'Source_Small_Sample_97.7%_AE'}
EVENT_COLUMNS = ['time_s', 'energy', 'rise_time', 'amplitude_dB']
STATE_ALIASES = {'no ae': 'No AE', 'raw': 'Raw AE', 'raw ae': 'Raw AE', '84.80%': '84.80% removed', '84.80% removed': '84.80% removed', '91.15%': '91.15% removed', '91.15% removed': '91.15% removed', '97.73%': '97.73% removed', '97.73% removed': '97.73% removed'}
ABLATION_GROUPS = {'All AE descriptors': ['Mutation Point', 'D_early', 'Delta_p'], 'Mutation Point': ['Mutation Point'], 'D_early': ['D_early'], 'Delta_p': ['Delta_p'], 'Maximum stress': ['Max Stress (MPa)'], 'Temperature': ['Temperature (C)'], 'Stress ratio': ['Stress Ratio'], 'Orientation': ['Orientation_001', 'Orientation_011', 'Orientation_111']}

class DataContractError(RuntimeError):
    pass

def canonical_state(value):
    key = str(value).strip().casefold()
    if key not in STATE_ALIASES:
        raise DataContractError(f'Unknown AE state label: {value!r}')
    return STATE_ALIASES[key]

def require_columns(frame, required, context):
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataContractError(f'{context} is missing required columns: {missing}')

def coerce_finite_numeric(frame, columns, context):
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors='coerce')
    invalid = result[columns].isna() | ~np.isfinite(result[columns])
    if invalid.any().any():
        counts = invalid.sum()
        bad = {column: int(count) for column, count in counts.items() if count}
        raise DataContractError(f'{context} contains missing or non-finite numeric values: {bad}')
    return result

def make_model(name, random_state):
    if name == 'XGBoost':
        return XGBRegressor(n_estimators=120, max_depth=2, learning_rate=0.035, subsample=0.9, colsample_bytree=0.9, min_child_weight=1.0, reg_alpha=0.02, reg_lambda=2.0, objective='reg:squarederror', random_state=random_state, n_jobs=1, verbosity=0)
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(n_estimators=160, max_depth=6, min_samples_leaf=1, max_features=0.9, random_state=random_state, n_jobs=1)
    if name == 'RandomForest':
        return RandomForestRegressor(n_estimators=160, max_depth=6, min_samples_leaf=1, max_features=0.9, random_state=random_state, n_jobs=1)
    return GradientBoostingRegressor(n_estimators=120, learning_rate=0.035, max_depth=2, min_samples_leaf=1, subsample=0.9, loss='huber', random_state=random_state)

def load_source(data_path):
    source = pd.read_excel(data_path, sheet_name=SOURCE_SHEET)
    required = ['Specimen ID', *SHARED_COLUMNS, *AE_COLUMNS, 'log10_Nf']
    require_columns(source, required, SOURCE_SHEET)
    source = coerce_finite_numeric(source, ['Temperature (C)', 'Max Stress (MPa)', 'Stress Ratio', *AE_COLUMNS, 'log10_Nf'], SOURCE_SHEET)
    source = source.reset_index(drop=True)
    if len(source) != 28:
        raise DataContractError(f'Expected 28 unrafted specimens in {SOURCE_SHEET}, found {len(source)}')
    source.insert(0, 'Sample index', np.arange(1, len(source) + 1))
    return source

def normalized_order(length):
    return np.zeros(length, dtype=float) if length <= 1 else np.arange(length, dtype=float) / (length - 1)

def segmented_change_point(energy):
    energy = np.asarray(energy, dtype=float)
    if energy.size < 10 or not np.isfinite(energy).all() or energy.sum() <= 0:
        raise DataContractError('An AE event sequence needs at least 10 finite events and positive total energy')
    u_full = normalized_order(energy.size)
    cumulative = np.cumsum(energy) / energy.sum()
    eligible = np.flatnonzero(u_full >= 0.2)
    if eligible.size > 800:
        eligible = eligible[np.linspace(0, eligible.size - 1, 800).astype(int)]
    u = u_full[eligible]
    y = cumulative[eligible]
    best = None
    for point in np.round(np.arange(0.3, 0.701, 0.005), 3):
        design = np.column_stack((np.ones_like(u), u, np.maximum(0.0, u - point)))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ coefficients
        sse = max(float(residual @ residual), np.finfo(float).tiny)
        bic = float(u.size * np.log(sse / u.size) + 3.0 * np.log(u.size))
        if best is None or bic < best[0]:
            best = (bic, point)
    return float(best[1])

def event_descriptor(frame):
    require_columns(frame, EVENT_COLUMNS, 'AE event table')
    ordered = coerce_finite_numeric(frame, EVENT_COLUMNS, 'AE event table').sort_values('time_s', kind='mergesort')
    energy = ordered['energy'].to_numpy(float)
    amplitude = ordered['amplitude_dB'].to_numpy(float)
    rise_time = ordered['rise_time'].to_numpy(float)
    u = normalized_order(len(ordered))
    cumulative = np.cumsum(energy) / energy.sum()
    point = segmented_change_point(energy)
    early_index = max(0, np.searchsorted(u, 0.2, side='right') - 1)
    d_early = float(cumulative[early_index])
    ra = rise_time / np.power(10.0, (amplitude - 40.0) / 20.0)
    pre = ra[(u >= max(0.2, point - 0.15)) & (u < point)]
    post = ra[(u >= point) & (u <= min(0.8, point + 0.15))]
    if pre.size == 0 or post.size == 0:
        raise DataContractError('The detected change point leaves an empty pre- or post-change RA window')
    ranks = pd.Series(np.concatenate([post, pre])).rank(method='average').to_numpy(float)
    u_post = float(ranks[:post.size].sum() - post.size * (post.size + 1) / 2.0)
    delta = float(2.0 * u_post / (pre.size * post.size) - 1.0)
    return {'Mutation Point': point, 'D_early': d_early, 'Delta_p': delta}

def load_event_states(data_path):
    states = {}
    for state, sheet in EVENT_STATE_SHEETS.items():
        frame = pd.read_excel(data_path, sheet_name=sheet)
        require_columns(frame, EVENT_COLUMNS, sheet)
        if frame.empty:
            raise DataContractError(f'{sheet} is empty')
        states[state] = frame
    return states

def calculate_event_state_summary(data_path):
    states = load_event_states(data_path)
    raw_count = len(states['Raw AE'])
    rows = []
    for state in ML_STATE_ORDER[1:]:
        frame = states[state]
        retention = len(frame) / raw_count
        rows.append({'AE state': state, 'Event count': len(frame), 'Retention fraction': retention, 'Removal percent': 100.0 * (1.0 - retention), **event_descriptor(frame)})
    return pd.DataFrame(rows)

def load_specimen_state_descriptors(data_path, source):
    try:
        descriptors = pd.read_excel(data_path, sheet_name=SPECIMEN_STATE_SHEET)
    except ValueError:
        return None
    required = ['Sample index', 'AE state', *AE_COLUMNS]
    require_columns(descriptors, required, SPECIMEN_STATE_SHEET)
    descriptors = coerce_finite_numeric(descriptors, ['Sample index', *AE_COLUMNS], SPECIMEN_STATE_SHEET)
    if not np.equal(descriptors['Sample index'], np.rint(descriptors['Sample index'])).all():
        raise DataContractError(f'{SPECIMEN_STATE_SHEET} contains non-integer Sample index values')
    descriptors['Sample index'] = descriptors['Sample index'].astype(int)
    descriptors['AE state'] = descriptors['AE state'].map(canonical_state)
    if descriptors.duplicated(['Sample index', 'AE state']).any():
        duplicates = descriptors.loc[descriptors.duplicated(['Sample index', 'AE state'], keep=False), ['Sample index', 'AE state']]
        raise DataContractError(f"{SPECIMEN_STATE_SHEET} contains duplicate sample/state rows: {duplicates.to_dict('records')[:5]}")
    if 'Specimen ID' in descriptors.columns:
        observed_ids = descriptors[['Sample index', 'Specimen ID']].copy()
        observed_ids['Specimen ID'] = observed_ids['Specimen ID'].astype(str)
        conflicting = observed_ids.groupby('Sample index')['Specimen ID'].nunique().loc[lambda x: x > 1]
        if not conflicting.empty:
            raise DataContractError(f'{SPECIMEN_STATE_SHEET} assigns multiple Specimen IDs to Sample index {conflicting.index.tolist()[:5]}')
        observed_ids = observed_ids.drop_duplicates('Sample index')
        expected_ids = source[['Sample index', 'Specimen ID']].copy()
        expected_ids['Specimen ID'] = expected_ids['Specimen ID'].astype(str)
        checked = observed_ids.merge(expected_ids, on='Sample index', how='inner', suffixes=('_observed', '_expected'))
        mismatch = checked.loc[checked['Specimen ID_observed'].ne(checked['Specimen ID_expected']), 'Sample index']
        if not mismatch.empty:
            raise DataContractError(f'{SPECIMEN_STATE_SHEET} Specimen ID does not match {SOURCE_SHEET} at Sample index {mismatch.tolist()[:5]}')
    return descriptors

def measured_state_table(source, descriptors, state):
    rows = descriptors.loc[descriptors['AE state'].eq(state), ['Sample index', *AE_COLUMNS]].copy()
    expected = source['Sample index'].tolist()
    observed = rows['Sample index'].tolist()
    if len(rows) != len(source) or set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise DataContractError(f'{SPECIMEN_STATE_SHEET} must contain one row for each of the {len(source)} samples in {state}; found {len(rows)}, missing={missing[:5]}, extra={extra[:5]}')
    base = source.drop(columns=AE_COLUMNS)
    return base.merge(rows, on='Sample index', how='left', validate='one_to_one', sort=False)

def build_states(data_path, required_states=None):
    requested = ML_STATE_ORDER if required_states is None else [canonical_state(state) for state in required_states]
    requested = [state for state in ML_STATE_ORDER if state in set(requested)]
    source = load_source(data_path)
    tables = {}
    if 'No AE' in requested:
        tables['No AE'] = source.drop(columns=AE_COLUMNS)
    for state in [value for value in requested if value != 'No AE']:
        sheet = SPECIMEN_STATE_SHEETS[state]
        table = pd.read_excel(data_path, sheet_name=sheet)
        required = ['Specimen ID', *SHARED_COLUMNS, *AE_COLUMNS, 'log10_Nf']
        require_columns(table, required, sheet)
        table = coerce_finite_numeric(table, ['Temperature (C)', 'Max Stress (MPa)', 'Stress Ratio', *AE_COLUMNS, 'log10_Nf'], sheet).reset_index(drop=True)
        if len(table) != len(source):
            raise DataContractError(f'Expected {len(source)} specimens in {sheet}, found {len(table)}')
        if not np.array_equal(table['Specimen ID'].astype(str), source['Specimen ID'].astype(str)):
            raise DataContractError(f'Specimen ordering in {sheet} does not match {SOURCE_SHEET}')
        if not np.allclose(table['log10_Nf'], source['log10_Nf'], rtol=0.0, atol=0.0):
            raise DataContractError(f'Fatigue-life targets in {sheet} do not match {SOURCE_SHEET}')
        table.insert(0, 'Sample index', np.arange(1, len(table) + 1))
        tables[state] = table
    return {state: tables[state] for state in requested}

def matrix(table, state):
    columns = SHARED_COLUMNS + ([] if state == 'No AE' else AE_COLUMNS)
    data = pd.get_dummies(table[columns], columns=['Orientation'], prefix='Orientation', dtype=float)
    data = data.rename(columns={'Orientation_[001]': 'Orientation_001', 'Orientation_[011]': 'Orientation_011', 'Orientation_[111]': 'Orientation_111'})
    for column in ('Orientation_001', 'Orientation_011', 'Orientation_111'):
        if column not in data:
            data[column] = 0.0
    order = ['Temperature (C)', 'Max Stress (MPa)', 'Stress Ratio', 'Orientation_001', 'Orientation_011', 'Orientation_111']
    if state != 'No AE':
        order += AE_COLUMNS
    return data[order].astype(float)

def metrics(y, prediction):
    return {'R2': float(r2_score(y, prediction)), 'RMSE_log10': float(math.sqrt(mean_squared_error(y, prediction))), 'MAE_log10': float(mean_absolute_error(y, prediction))}

def validate_aligned_tables(tables):
    state_order = [state for state in ML_STATE_ORDER if state in tables]
    reference = tables[state_order[0]]
    reference_index = reference['Sample index'].to_numpy(int)
    reference_y = reference['log10_Nf'].to_numpy(float)
    for state in state_order[1:]:
        table = tables[state]
        if not np.array_equal(reference_index, table['Sample index'].to_numpy(int)):
            raise DataContractError(f'Sample ordering in {state} does not match the reference state')
        if not np.allclose(reference_y, table['log10_Nf'].to_numpy(float), rtol=0.0, atol=0.0):
            raise DataContractError(f'Fatigue-life targets in {state} do not match the reference state')
    return (state_order, reference_y)

def repeated_cv(tables):
    state_order, y = validate_aligned_tables(tables)
    metric_rows = []
    prediction_rows = []
    for repeat, seed in enumerate(REPEAT_SEEDS, start=1):
        folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(np.arange(len(y))))
        for state in state_order:
            x = matrix(tables[state], state)
            for model_name in MODEL_ORDER:
                prediction = np.full(len(y), np.nan)
                fold_ids = np.zeros(len(y), dtype=int)
                for fold, (train, test) in enumerate(folds, start=1):
                    model = make_model(model_name, seed + fold * 1009)
                    model.fit(x.iloc[train], y[train])
                    prediction[test] = model.predict(x.iloc[test])
                    fold_ids[test] = fold
                metric_rows.append({'Repeat': repeat, 'Seed': seed, 'AE state': state, 'Model': model_name, **metrics(y, prediction)})
                for i in range(len(y)):
                    prediction_rows.append({'Repeat': repeat, 'Seed': seed, 'Fold': int(fold_ids[i]), 'AE state': state, 'Model': model_name, 'Sample index': int(tables[state].iloc[i]['Sample index']), 'Specimen ID': str(tables[state].iloc[i]['Specimen ID']), 'Observed log10_Nf': y[i], 'Predicted log10_Nf': prediction[i], 'Residual log10_Nf': y[i] - prediction[i]})
    repeat_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    consensus = predictions.groupby(['AE state', 'Model', 'Sample index', 'Specimen ID', 'Observed log10_Nf'], as_index=False, sort=False)['Predicted log10_Nf'].mean().rename(columns={'Predicted log10_Nf': 'Consensus predicted log10_Nf'})
    consensus['Absolute error log10'] = np.abs(consensus['Observed log10_Nf'] - consensus['Consensus predicted log10_Nf'])
    summary_rows = []
    for state in state_order:
        for model_name in MODEL_ORDER:
            repeats = repeat_metrics.loc[repeat_metrics['AE state'].eq(state) & repeat_metrics['Model'].eq(model_name)]
            block = consensus.loc[consensus['AE state'].eq(state) & consensus['Model'].eq(model_name)]
            score = metrics(block['Observed log10_Nf'].to_numpy(), block['Consensus predicted log10_Nf'].to_numpy())
            summary_rows.append({'AE state': state, 'Model': model_name, 'Consensus R2': score['R2'], 'Consensus RMSE_log10': score['RMSE_log10'], 'Consensus MAE_log10': score['MAE_log10'], 'Median repeat R2': repeats['R2'].median(), 'R2 2.5%': repeats['R2'].quantile(0.025), 'R2 97.5%': repeats['R2'].quantile(0.975), 'Median repeat RMSE_log10': repeats['RMSE_log10'].median()})
    summary = pd.DataFrame(summary_rows)
    ensemble = consensus.groupby(['AE state', 'Sample index', 'Specimen ID', 'Observed log10_Nf'], as_index=False, sort=False)['Consensus predicted log10_Nf'].mean().rename(columns={'Consensus predicted log10_Nf': 'Four-model ensemble predicted log10_Nf'})
    ensemble['Absolute error log10'] = np.abs(ensemble['Observed log10_Nf'] - ensemble['Four-model ensemble predicted log10_Nf'])
    return (repeat_metrics, predictions, consensus, summary, ensemble)

def ablation_task(repeat, seed, group, model_name, x, y, folds, full_r2):
    prediction = np.full(len(y), np.nan)
    for fold, (train, test) in enumerate(folds, start=1):
        model = make_model(model_name, seed + fold * 1009)
        model.fit(x.iloc[train], y[train])
        prediction[test] = model.predict(x.iloc[test])
    ablated = float(r2_score(y, prediction))
    return {'Repeat': repeat, 'Seed': seed, 'Model': model_name, 'Feature group removed': group, 'Full R2': full_r2, 'Ablated R2': ablated, 'R2 loss after removal': full_r2 - ablated}

def ablation_summary(tables, repeat_metrics):
    state = '91.15% removed'
    y = tables[state]['log10_Nf'].to_numpy(float)
    x_full = matrix(tables[state], state)
    full_lookup = repeat_metrics.loc[repeat_metrics['AE state'].eq(state)].set_index(['Repeat', 'Model'])['R2']
    no_ae = repeat_metrics.loc[repeat_metrics['AE state'].eq('No AE')].set_index(['Repeat', 'Model'])['R2']
    rows = []
    for repeat, seed in enumerate(REPEAT_SEEDS, start=1):
        for model_name in MODEL_ORDER:
            full = float(full_lookup.loc[repeat, model_name])
            base = float(no_ae.loc[repeat, model_name])
            rows.append({'Repeat': repeat, 'Seed': seed, 'Model': model_name, 'Feature group removed': 'All AE descriptors', 'Full R2': full, 'Ablated R2': base, 'R2 loss after removal': full - base})
        folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(np.arange(len(y))))
        jobs = []
        for group, columns in ABLATION_GROUPS.items():
            if group == 'All AE descriptors':
                continue
            for model_name in MODEL_ORDER:
                jobs.append(delayed(ablation_task)(repeat, seed, group, model_name, x_full.drop(columns=columns), y, folds, float(full_lookup.loc[repeat, model_name])))
        rows.extend(Parallel(n_jobs=-1, backend='loky')(jobs))
    detail = pd.DataFrame(rows)
    return detail.groupby(['Feature group removed', 'Model'], as_index=False, sort=False).agg(**{'Median R2 loss': ('R2 loss after removal', 'median'), 'Mean R2 loss': ('R2 loss after removal', 'mean'), 'R2 loss 2.5%': ('R2 loss after removal', lambda x: x.quantile(0.025)), 'R2 loss 97.5%': ('R2 loss after removal', lambda x: x.quantile(0.975)), 'Positive fraction': ('R2 loss after removal', lambda x: np.mean(np.asarray(x) > 0))})

def grouped_condition_sensitivity(data_path):
    source = load_source(data_path)
    target = source['log10_Nf'].to_numpy(float)
    groups = pd.factorize(pd.MultiIndex.from_frame(source[['Temperature (C)', 'Max Stress (MPa)', 'Stress Ratio', 'Orientation']]), sort=True)[0]
    tables = {'No AE': matrix(source, 'No AE'), '91.15% removed': matrix(source, '91.15% removed')}
    split_sets = {'Leave-one-condition-group-out': list(LeaveOneGroupOut().split(tables['No AE'], target, groups)), '50 condition-group shuffles': list(GroupShuffleSplit(n_splits=50, test_size=0.2, random_state=20260813).split(tables['No AE'], target, groups))}
    metric_rows, prediction_rows = ([], [])
    for family, splits in split_sets.items():
        for state, features in tables.items():
            values = features.to_numpy(float)
            for model_name in MODEL_ORDER:
                for split, (train, test) in enumerate(splits, start=1):
                    model = make_model(model_name, 1000 + split)
                    model.fit(values[train], target[train])
                    prediction = model.predict(values[test])
                    metric_rows.append({'Split family': family, 'AE state': state, 'Model': model_name, 'Split': split, 'R2': r2_score(target[test], prediction) if len(test) > 1 else np.nan, 'MAE_log10': mean_absolute_error(target[test], prediction), 'Test records': len(test)})
                    for position, row in enumerate(test):
                        prediction_rows.append({'Split family': family, 'AE state': state, 'Model': model_name, 'Split': split, 'Sample index': int(source.iloc[row]['Sample index']), 'Condition group': int(groups[row]) + 1, 'Observed log10_Nf': target[row], 'Predicted log10_Nf': prediction[position]})
    return (pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows))

def calculate_panel(panel, data_path):
    panel_states = {'3a': 'No AE', '3b': 'Raw AE', '3c': '84.80% removed', '3d': '91.15% removed', '3e': '97.73% removed'}
    if panel in panel_states:
        target = panel_states[panel]
        required = ['No AE'] if target == 'No AE' else ['No AE', target]
        tables = build_states(data_path, required)
        repeat_metrics, _, consensus, summary, _ = repeated_cv(tables)
        result = consensus.loc[consensus['AE state'].eq(target)].copy()
        return result.merge(summary, on=['AE state', 'Model'], how='left')
    if panel in {'3f', '3g', '3h'}:
        tables = build_states(data_path, ML_STATE_ORDER)
        repeat_metrics, _, _, summary, ensemble = repeated_cv(tables)
        if panel == '3f':
            detail = repeat_metrics[['Repeat', 'Seed', 'AE state', 'Model', 'RMSE_log10']].copy()
            detail['MSE_log10'] = np.square(detail['RMSE_log10'])
            result = detail.groupby(['AE state', 'Model'], as_index=False, sort=False).agg(**{'Median repeat MSE_log10': ('MSE_log10', 'median'), 'Mean repeat MSE_log10': ('MSE_log10', 'mean'), 'MSE 2.5%': ('MSE_log10', lambda x: x.quantile(0.025)), 'MSE 97.5%': ('MSE_log10', lambda x: x.quantile(0.975))})
            return result.merge(summary[['AE state', 'Model', 'Consensus RMSE_log10']], on=['AE state', 'Model'])
        if panel == '3g':
            baseline = repeat_metrics.loc[repeat_metrics['AE state'].eq('No AE'), ['Repeat', 'Seed', 'Model', 'R2']].rename(columns={'R2': 'No AE R2'})
            paired = repeat_metrics.loc[~repeat_metrics['AE state'].eq('No AE'), ['Repeat', 'Seed', 'AE state', 'Model', 'R2']].rename(columns={'R2': 'State R2'}).merge(baseline, on=['Repeat', 'Seed', 'Model'])
            paired['Delta R2 vs No AE'] = paired['State R2'] - paired['No AE R2']
            return paired.groupby(['AE state', 'Model'], as_index=False, sort=False).agg(**{'Median Delta R2': ('Delta R2 vs No AE', 'median'), 'Mean Delta R2': ('Delta R2 vs No AE', 'mean'), 'Delta R2 2.5%': ('Delta R2 vs No AE', lambda x: x.quantile(0.025)), 'Delta R2 97.5%': ('Delta R2 vs No AE', lambda x: x.quantile(0.975)), 'Positive fraction': ('Delta R2 vs No AE', lambda x: np.mean(np.asarray(x) > 0))})
        return ensemble
    if panel == '3i':
        tables = build_states(data_path, ['No AE', '91.15% removed'])
        repeat_metrics, _, _, _, _ = repeated_cv(tables)
        return ablation_summary(tables, repeat_metrics)
    raise ValueError(f'Unknown Figure 3 panel: {panel!r}')

def main():
    parser = argparse.ArgumentParser(description='Export one Figure 3 calculation table.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Input workbook (default: repository data/Supplementary_Data.xlsx).')
    parser.add_argument('--panel', choices=[f'3{x}' for x in 'abcdefghi'], required=True)
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
