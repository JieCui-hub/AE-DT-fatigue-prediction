"""Current Figure 6 condition-reweighting and transfer calculations."""

from __future__ import annotations
import argparse
import math
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPOSITORY_ROOT / 'data' / 'Supplementary_Data.xlsx'
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
AE_COLUMNS = ['Mutation Point', 'D_early', 'Delta_p']
MS_COLUMNS = ['gamma_prime_volume_fraction_pct', 'average_gamma_prime_edge_length_um', 'average_gamma_channel_width_um']
SOURCE_CONDITIONS = ['Temperature (C)', 'Max Stress (MPa)', 'Stress Ratio']
TARGET_CONDITIONS = ['temperature_C', 'sigma_max_MPa', 'stress_ratio_R', 'orientation_family_001', 'orientation_family_011', 'orientation_family_111', 'orientation_family_other']
CONDITION_LABELS = ['Temperature', 'Max stress', 'Stress ratio', 'Orientation']
N_BOOTSTRAP = 10000
CURRENT_PANELS = tuple(f'6{x}' for x in 'abcdefghi')
LEGACY_PANEL_ALIASES = {
    '5b': '6b',
    '5c': '6g',
    '5d': '6e',
    '5e': '6a',
    '5f': '6f',
    '5g': '6c',
    '5h': '6d',
    '5i': '6h',
    '5j': '6i',
}

def source_design(data, descriptors=None):
    frame = pd.get_dummies(data[SOURCE_CONDITIONS + ['Orientation']], columns=['Orientation'], prefix='Orientation', dtype=float)
    if descriptors is not None:
        frame = pd.concat([frame, data[descriptors].reset_index(drop=True)], axis=1)
    return frame

def source_group_columns(columns, label):
    if label == 'Temperature':
        return ['Temperature (C)']
    if label == 'Max stress':
        return ['Max Stress (MPa)']
    if label == 'Stress ratio':
        return ['Stress Ratio']
    return [column for column in columns if str(column).startswith('Orientation_')]

def source_model(seed):
    return make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.85, random_state=seed, n_jobs=1))

def condition_shares(model, x_test, y_test, seed):
    baseline = mean_squared_error(y_test, model.predict(x_test))
    rng = np.random.default_rng(seed)
    importance = []
    for label in CONDITION_LABELS:
        columns = source_group_columns(x_test.columns, label)
        changes = []
        for _ in range(10):
            permuted = x_test.copy()
            order = rng.permutation(len(permuted))
            permuted.loc[:, columns] = permuted.loc[:, columns].iloc[order].to_numpy()
            changes.append(mean_squared_error(y_test, model.predict(permuted)) - baseline)
        importance.append(np.mean(changes))
    positive = np.maximum(np.asarray(importance), 0.0)
    return np.full(4, np.nan) if positive.sum() == 0 else positive / positive.sum()

def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    return np.quantile(values[index].mean(axis=1), [0.025, 0.975])

def learn_reweighting(source, descriptors, label):
    y = source['log10_Nf'].to_numpy(float)
    groups = source[SOURCE_CONDITIONS + ['Orientation']].astype(str).agg('|'.join, axis=1).to_numpy()
    base = source_design(source)
    enhanced = source_design(source, descriptors)
    splits = GroupShuffleSplit(n_splits=50, test_size=0.2, random_state=20260726).split(base, y, groups)
    rows = []
    for split, (train, test) in enumerate(splits):
        for state, features, seed, offset in (('Base', base, 1000 + split, 0), (label, enhanced, 5000 + split, 1)):
            model = source_model(seed)
            model.fit(features.iloc[train], y[train])
            prediction = model.predict(features.iloc[test])
            shares = condition_shares(model, features.iloc[test], y[test], 100000 + split * 31 + offset)
            row = {'Split': split + 1, 'State': state, 'Test R2': r2_score(y[test], prediction)}
            row.update(dict(zip(CONDITION_LABELS, shares)))
            rows.append(row)
    detail = pd.DataFrame(rows)
    base_detail = detail.loc[detail['State'].eq('Base')].set_index('Split')
    enhanced_detail = detail.loc[detail['State'].eq(label)].set_index('Split')
    summary_rows = []
    for position, condition in enumerate(CONDITION_LABELS):
        base_values = base_detail[condition].to_numpy(float)
        enhanced_values = enhanced_detail[condition].to_numpy(float)
        valid = np.isfinite(base_values) & np.isfinite(enhanced_values)
        delta = enhanced_values[valid] - base_values[valid]
        low, high = bootstrap_mean_ci(delta, 20260727 + position)
        summary_rows.append({'Condition': condition, 'Base share mean': np.nanmean(base_values), 'Enhanced share mean': np.nanmean(enhanced_values), 'Paired base share mean': np.mean(base_values[valid]), 'Paired enhanced share mean': np.mean(enhanced_values[valid]), 'Share change': np.mean(delta), 'CI95 low': low, 'CI95 high': high, 'Valid paired splits': int(valid.sum())})
    summary = pd.DataFrame(summary_rows)
    base_weights = summary['Base share mean'].to_numpy(float)
    base_weights = base_weights / base_weights.sum()
    enhanced_weights = summary['Enhanced share mean'].to_numpy(float)
    enhanced_weights = enhanced_weights / enhanced_weights.sum()
    summary['Base transfer weight'] = base_weights
    summary['Enhanced transfer weight'] = enhanced_weights
    return (detail, summary, base_weights, enhanced_weights)

def ae_teacher_matrix(data):
    values = data[AE_COLUMNS].to_numpy(float)
    scaled = np.column_stack(((values[:, 0] - 0.18) / 0.64, (values[:, 1] - 0.08) / 0.68, (values[:, 2] - 0.1) / 0.62))
    scaled = np.clip(scaled, 1e-06, 1.0 - 1e-06)
    return np.log(scaled / (1.0 - scaled))

def teacher_matrix(data, modality):
    if modality == 'Conditions':
        return source_design(data).to_numpy(float)
    if modality == 'MS':
        return data[MS_COLUMNS].to_numpy(float)
    if modality == 'AE':
        return ae_teacher_matrix(data)
    return np.column_stack((ae_teacher_matrix(data), data[MS_COLUMNS].to_numpy(float)))

def teacher_metrics(source):
    y = source['log10_Nf'].to_numpy(float)
    groups = source[SOURCE_CONDITIONS + ['Orientation']].astype(str).agg('|'.join, axis=1).to_numpy()
    rows = []
    for modality in ('Conditions', 'MS', 'AE', 'AE + MS'):
        x = teacher_matrix(source, modality)
        prediction = np.zeros(len(source))
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for train, test in splitter.split(x, y, groups):
            model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=0.05))
            model.fit(x[train], y[train])
            prediction[test] = model.predict(x[test])
        rows.append({'Modality': modality, 'Grouped OOF R2': r2_score(y, prediction), 'Grouped OOF RMSE': math.sqrt(mean_squared_error(y, prediction)), 'Grouped OOF MAE': mean_absolute_error(y, prediction)})
    return pd.DataFrame(rows)

def literature_groups(literature):
    return pd.factorize(pd.MultiIndex.from_frame(literature[TARGET_CONDITIONS]), sort=True)[0]

def target_prior(features, target, train, test, weights):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(features.iloc[train][TARGET_CONDITIONS]))
    x_test = scaler.transform(imputer.transform(features.iloc[test][TARGET_CONDITIONS]))
    dimensions = np.array([weights[0], weights[1], weights[2], weights[3] / 4.0, weights[3] / 4.0, weights[3] / 4.0, weights[3] / 4.0])
    scale = np.sqrt(np.maximum(dimensions, 1e-08))
    model = SVR(C=30.0, gamma=0.3, epsilon=0.08, kernel='rbf')
    model.fit(x_train * scale, target[train])
    return model.predict(x_test * scale)

def group_bootstrap_delta_ci(actual, baseline, prediction, groups, rng):
    actual = np.asarray(actual, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    group_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    delta = np.full(N_BOOTSTRAP, np.nan)
    for iteration in range(N_BOOTSTRAP):
        selected = rng.integers(0, len(group_indices), size=len(group_indices))
        index = np.concatenate([group_indices[position] for position in selected])
        if np.ptp(actual[index]) == 0:
            continue
        delta[iteration] = r2_score(actual[index], prediction[index]) - r2_score(actual[index], baseline[index])
    return np.nanquantile(delta, [0.025, 0.975])

def target_results(literature, base_weights, ae_weights, ms_weights):
    features = literature[TARGET_CONDITIONS]
    target = literature['log10_Nf'].to_numpy(float)
    groups = literature_groups(literature)
    split_sets = {'repeated_group': list(GroupShuffleSplit(n_splits=50, test_size=0.2, random_state=20260717).split(features, target, groups)), 'logo': list(LeaveOneGroupOut().split(features, target, groups))}
    metric_rows = []
    prediction_rows = []
    for family, splits in split_sets.items():
        for seed, (train, test) in enumerate(splits):
            baseline = target_prior(features, target, train, test, base_weights)
            ae_prior = target_prior(features, target, train, test, ae_weights)
            ms_prior = target_prior(features, target, train, test, ms_weights)
            predictions = {'Base': baseline, 'MS': ms_prior, 'AE': ae_prior}
            held_out_groups = np.unique(groups[test])
            for method, prediction in predictions.items():
                holdout_r2 = r2_score(target[test], prediction) if len(test) >= 2 and np.ptp(target[test]) > 0 else np.nan
                metric_rows.append({'Split family': family, 'Seed': seed + 1, 'Method': method, 'Held-out groups': len(held_out_groups), 'Test records': len(test), 'Holdout R2': holdout_r2, 'Holdout RMSE': math.sqrt(mean_squared_error(target[test], prediction)), 'Holdout MAE': mean_absolute_error(target[test], prediction)})
                for position, row_index in enumerate(test):
                    prediction_rows.append({'Split family': family, 'Seed': seed + 1, 'Method': method, 'Condition group': int(groups[row_index]) + 1, 'Row index': int(row_index), 'Original row index': int(literature.iloc[row_index]['original_row_index']), 'Measured log10 Nf': target[row_index], 'Predicted log10 Nf': prediction[position]})
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    consensus = predictions.groupby(['Split family', 'Method', 'Row index'], as_index=False).agg(**{'Condition group': ('Condition group', 'first'), 'Original row index': ('Original row index', 'first'), 'Measured log10 Nf': ('Measured log10 Nf', 'first'), 'Predicted log10 Nf': ('Predicted log10 Nf', 'mean'), 'Prediction SD': ('Predicted log10 Nf', 'std'), 'Test appearances': ('Seed', 'size')})
    effect_rows = []
    for method in ('MS', 'AE'):
        rng = np.random.default_rng(20260719)
        for family in ('repeated_group', 'logo'):
            frame = consensus.loc[consensus['Split family'].eq(family)]
            wide = frame.pivot(index='Row index', columns='Method', values='Predicted log10 Nf')
            actual = frame.drop_duplicates('Row index').set_index('Row index').loc[wide.index, 'Measured log10 Nf'].to_numpy(float)
            effect_groups = frame.drop_duplicates('Row index').set_index('Row index').loc[wide.index, 'Condition group'].to_numpy(int)
            baseline = wide['Base'].to_numpy(float)
            prediction = wide[method].to_numpy(float)
            interval_rng = np.random.default_rng(20260802) if method == 'AE' else rng
            low, high = group_bootstrap_delta_ci(actual, baseline, prediction, effect_groups, interval_rng)
            base_r2 = r2_score(actual, baseline)
            score = r2_score(actual, prediction)
            effect_rows.append({'Split family': family, 'Method': method, 'Baseline consensus R2': base_r2, 'Consensus R2': score, 'Delta R2': score - base_r2, 'CI95 low': low, 'CI95 high': high, 'Unique records': len(actual)})
    return (metrics, predictions, consensus, pd.DataFrame(effect_rows))

def build_results(data_path):
    source = pd.read_excel(data_path, sheet_name='Source_Small_Sample_91.2%_AE').reset_index(drop=True)
    if len(source) != 28:
        raise ValueError(f'Expected 28 source specimens, found {len(source)}')
    literature = pd.read_excel(data_path, sheet_name='Literature_Large_Sample')
    rename_columns = {}
    if 'ID' in literature.columns and 'original_row_index' not in literature.columns:
        rename_columns['ID'] = 'original_row_index'
    if 'DOI' in literature.columns and 'doi' not in literature.columns:
        rename_columns['DOI'] = 'doi'
    literature = literature.rename(columns=rename_columns)
    if 'original_row_index' not in literature.columns:
        literature.insert(0, 'original_row_index', np.arange(1, len(literature) + 1))
    required_columns = set(TARGET_CONDITIONS + ['original_row_index', 'doi', 'log10_Nf'])
    missing_columns = sorted(required_columns.difference(literature.columns))
    if missing_columns:
        raise ValueError(f'Literature_Large_Sample is missing required columns: {missing_columns}')
    if literature['original_row_index'].isna().any() or literature['original_row_index'].duplicated().any():
        raise ValueError('Literature_Large_Sample requires unique, non-missing ID values')
    _, ae_reweighting, base_weights, ae_weights = learn_reweighting(source, AE_COLUMNS, 'Base + AE')
    _, ms_reweighting, ms_base_weights, ms_weights = learn_reweighting(source, MS_COLUMNS, 'Base + MS')
    if not np.allclose(base_weights, ms_base_weights, rtol=0.0, atol=1e-12):
        raise RuntimeError('Base condition weights changed between the paired AE and MS calculations')
    teachers = teacher_metrics(source)
    metrics, predictions, consensus, effects = target_results(literature, base_weights, ae_weights, ms_weights)
    return (teachers, ae_reweighting, ms_reweighting, metrics, consensus, effects)

def figure_6b(effects):
    rows = []
    for family in ('repeated_group', 'logo'):
        block = effects.loc[effects['Split family'].eq(family)]
        rows.append({'Split family': family, 'Method': 'Base', 'Consensus R2': block['Baseline consensus R2'].iloc[0]})
        for method in ('MS', 'AE'):
            rows.append({'Split family': family, 'Method': method, 'Consensus R2': block.loc[block['Method'].eq(method), 'Consensus R2'].iloc[0]})
    return pd.DataFrame(rows)

def figure_6g(consensus):
    frame = consensus.loc[consensus['Split family'].eq('repeated_group'), ['Split family', 'Method', 'Condition group', 'Row index', 'Measured log10 Nf', 'Predicted log10 Nf']].copy()
    frame['Absolute error'] = np.abs(frame['Predicted log10 Nf'] - frame['Measured log10 Nf'])
    return frame.reset_index(drop=True)

def figure_6e(metrics):
    wide = metrics.loc[metrics['Split family'].eq('repeated_group')].pivot(index=['Split family', 'Seed'], columns='Method', values='Holdout R2').reset_index()
    wide['AE minus Base R2'] = wide['AE'] - wide['Base']
    wide['MS minus Base R2'] = wide['MS'] - wide['Base']
    return wide.rename(columns={'AE': 'AE R2', 'MS': 'MS R2', 'Base': 'Base R2'})

def figure_6f(consensus):
    return consensus.loc[consensus['Method'].isin(['Base', 'AE']), ['Split family', 'Method', 'Condition group', 'Row index', 'Measured log10 Nf', 'Predicted log10 Nf']].reset_index(drop=True)

def figure_6c(metrics):
    paired = figure_6e(metrics)
    paired['AE minus MS R2'] = paired['AE R2'] - paired['MS R2']
    return paired[['Split family', 'Seed', 'MS R2', 'AE R2', 'AE minus MS R2']].copy()

def figure_6i(consensus):
    frame = consensus.loc[consensus['Split family'].eq('logo')]
    wide = frame.pivot(index='Row index', columns='Method', values='Predicted log10 Nf')
    actual = frame.drop_duplicates('Row index').set_index('Row index').loc[wide.index, 'Measured log10 Nf']
    result = pd.DataFrame({'Row index': wide.index.to_numpy(int), 'Measured log10 Nf': actual.to_numpy(float), 'Baseline prediction': wide['Base'].to_numpy(float), 'AE prediction': wide['AE'].to_numpy(float)})
    result['Baseline residual'] = result['Measured log10 Nf'] - result['Baseline prediction']
    result['AE prediction update'] = result['AE prediction'] - result['Baseline prediction']
    result['Delta absolute error'] = np.abs(result['Baseline residual']) - np.abs(result['Measured log10 Nf'] - result['AE prediction'])
    result['Lower absolute error'] = result['Delta absolute error'] > 0
    result['Directionally aligned'] = result['Baseline residual'] * result['AE prediction update'] > 0
    result['Quantile bin'] = pd.qcut(result['Baseline residual'], q=7, labels=False, duplicates='drop').astype(int) + 1
    rng = np.random.default_rng(20260731)
    bin_rows = []
    for bin_id, block in result.groupby('Quantile bin'):
        values = block['AE prediction update'].to_numpy(float)
        bootstrap = np.array([np.median(rng.choice(values, len(values), replace=True)) for _ in range(1200)])
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        bin_rows.append({'Quantile bin': int(bin_id), 'Bin records': len(block), 'Baseline residual median': np.median(block['Baseline residual']), 'AE update median': np.median(values), 'AE update CI95 low': low, 'AE update CI95 high': high})
    return result.merge(pd.DataFrame(bin_rows), on='Quantile bin', how='left')

def figure_6h(ae_reweighting, ms_reweighting):
    ae = ae_reweighting.set_index('Condition')
    ms = ms_reweighting.set_index('Condition')
    conditions = CONDITION_LABELS
    base = ae.loc[conditions, 'Base transfer weight'].to_numpy(float)
    ms_base = ms.loc[conditions, 'Base transfer weight'].to_numpy(float)
    if not np.allclose(base, ms_base, rtol=0.0, atol=1e-12):
        raise RuntimeError('Base condition weights changed between the AE and microstructure calculations')
    return pd.DataFrame({
        'Condition': conditions,
        'Base weight': base,
        'AE-informed weight': ae.loc[conditions, 'Enhanced transfer weight'].to_numpy(float),
        'MS-informed weight': ms.loc[conditions, 'Enhanced transfer weight'].to_numpy(float),
    })

def calculate_panel(panel, data_path):
    panel = LEGACY_PANEL_ALIASES.get(panel, panel)
    if panel not in CURRENT_PANELS:
        choices = ', '.join(CURRENT_PANELS)
        raise ValueError(f'Unknown panel {panel!r}; expected one of: {choices}')
    teachers, ae_reweighting, ms_reweighting, metrics, consensus, effects = build_results(data_path)
    outputs = {
        '6a': lambda: teachers,
        '6b': lambda: figure_6b(effects),
        '6c': lambda: figure_6c(metrics),
        '6d': lambda: effects.reset_index(drop=True),
        '6e': lambda: figure_6e(metrics),
        '6f': lambda: figure_6f(consensus),
        '6g': lambda: figure_6g(consensus),
        '6h': lambda: figure_6h(ae_reweighting, ms_reweighting),
        '6i': lambda: figure_6i(consensus),
    }
    return outputs[panel]()

def main():
    parser = argparse.ArgumentParser(description='Export one Figure 6 condition-reweighting calculation table.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Input workbook (default: repository data/Supplementary_Data.xlsx).')
    parser.add_argument(
        '--panel',
        choices=[*CURRENT_PANELS, *LEGACY_PANEL_ALIASES],
        required=True,
        help='Current Figure 6 selector (6a-6i); legacy 5b-5j aliases remain accepted.',
    )
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
