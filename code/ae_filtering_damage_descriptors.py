"""Event-level AE filtering, trajectory-descriptor and quality calculations.

The module retains the event-level filtering, change-point, descriptor,
amplitude-completeness and filter-quality calculations used for the manuscript.
Run it as a script to export one panel table at a time.
"""

from __future__ import annotations
import argparse
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPOSITORY_ROOT / 'data' / 'Supplementary_Data.xlsx'
import numpy as np
import pandas as pd
from scipy import stats
STATE_SHEETS = {'RAW': 'AE_Raw', '84.80%': 'AE_Filter_84.8pct', '91.15%': 'AE_Filter_91.2pct', '97.73%': 'AE_Filter_97.7pct'}
STATE_ORDER = tuple(STATE_SHEETS)

def normalized_order(n):
    return np.zeros(n, dtype=float) if n <= 1 else np.arange(n, dtype=float) / (n - 1)

def load_states(data_path):
    return {name: pd.read_excel(data_path, sheet_name=sheet) for name, sheet in STATE_SHEETS.items()}

def segmented_fit(energy):
    energy = np.asarray(energy, dtype=float)
    u_full = normalized_order(energy.size)
    cumulative = np.cumsum(energy) / energy.sum()
    eligible = np.flatnonzero(u_full >= 0.2)
    if eligible.size > 800:
        eligible = eligible[np.linspace(0, eligible.size - 1, 800).astype(int)]
    u = u_full[eligible]
    y = cumulative[eligible]
    rows = []
    best = None
    for point in np.round(np.arange(0.3, 0.701, 0.005), 3):
        design = np.column_stack((np.ones_like(u), u, np.maximum(0.0, u - point)))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ coefficients
        sse = max(float(residual @ residual), np.finfo(float).tiny)
        bic = float(u.size * np.log(sse / u.size) + 3.0 * np.log(u.size))
        rows.append((point, bic))
        if best is None or bic < best[0]:
            best = (bic, point, coefficients)
    scan = pd.DataFrame(rows, columns=['candidate_u', 'BIC'])
    scan['Delta_BIC'] = scan['BIC'] - scan['BIC'].min()
    scan['log10_1_plus_Delta_BIC'] = np.log10(1.0 + scan['Delta_BIC'])
    fitted = np.column_stack((np.ones_like(u_full), u_full, np.maximum(0.0, u_full - best[1]))) @ best[2]
    return (float(best[1]), fitted, scan)

def descriptor(frame):
    frame = frame.sort_values('time_s', kind='mergesort').reset_index(drop=True)
    energy = frame['energy'].to_numpy(float)
    amplitude = frame['amplitude_dB'].to_numpy(float)
    rise_time = frame['rise_time'].to_numpy(float)
    u = normalized_order(len(frame))
    cumulative = np.cumsum(energy) / energy.sum()
    point = segmented_fit(energy)[0]
    d_early = float(cumulative[np.searchsorted(u, 0.2, side='right') - 1])
    ra = rise_time / np.power(10.0, (amplitude - 40.0) / 20.0)
    pre = ra[(u >= max(0.2, point - 0.15)) & (u < point)]
    post = ra[(u >= point) & (u <= min(0.8, point + 0.15))]
    statistic = stats.mannwhitneyu(post, pre, alternative='two-sided').statistic
    delta = float(2.0 * statistic / (pre.size * post.size) - 1.0)
    return (point, d_early, delta, pre, post)

def geometric_amplitude_fit(amplitude, completeness):
    values = np.asarray(amplitude, dtype=float)
    values = values[np.isfinite(values) & (values >= completeness)]
    gaps = values - completeness
    rounded = np.rint(gaps)
    if not np.allclose(gaps, rounded, rtol=0.0, atol=1e-9):
        raise ValueError('Geometric amplitude fitting requires integer dB excesses.')
    gaps = rounded.astype(int)
    if gaps.size < 3 or np.mean(gaps) <= 0.0:
        return (np.nan, np.nan, np.nan, int(gaps.size))
    mean_gap = float(np.mean(gaps))
    q_value = mean_gap / (1.0 + mean_gap)
    b_value = float(-20.0 * np.log10(q_value))
    unique, counts = np.unique(gaps, return_counts=True)
    survival = np.cumsum(counts[::-1])[::-1].astype(float) / gaps.size
    model = np.power(q_value, unique)
    distance = float(np.max(np.abs(survival - model)))
    return (q_value, b_value, distance, int(gaps.size))

def geometric_bootstrap_gof(amplitude, completeness, rng, n_bootstrap=2000):
    q_value, b_value, observed_distance, count = geometric_amplitude_fit(amplitude, completeness)
    if not np.isfinite(q_value):
        return (q_value, b_value, observed_distance, count, np.nan)
    simulated = rng.geometric(1.0 - q_value, size=(n_bootstrap, count)) - 1
    simulated_mean = np.mean(simulated, axis=1)
    simulated_q = simulated_mean / (1.0 + simulated_mean)
    ordered = np.sort(simulated, axis=1)
    first = np.column_stack((np.ones(n_bootstrap, dtype=bool), ordered[:, 1:] != ordered[:, :-1]))
    empirical = (count - np.arange(count, dtype=float)) / count
    distance = np.abs(empirical - np.power(simulated_q[:, None], ordered))
    bootstrap_distance = np.max(np.where(first, distance, -np.inf), axis=1)
    probability = float(np.mean(bootstrap_distance >= observed_distance))
    return (q_value, b_value, observed_distance, count, probability)

def quality_gate(frame):
    return (frame['amplitude_dB'].to_numpy(float) >= 50.0) & (frame['energy'].to_numpy(float) >= 5.0) & (frame['rise_time'].to_numpy(float) >= 40.0)

def descriptor_samples(frame, seed, n_subsamples=60):
    rng = np.random.default_rng(seed)
    size = max(200, int(round(0.8 * len(frame))))
    values = []
    for _ in range(n_subsamples):
        index = np.sort(rng.choice(len(frame), size=size, replace=False))
        values.append(descriptor(frame.iloc[index])[:3])
    return np.asarray(values, dtype=float)

def stability(samples):
    penalty = float(np.mean(np.std(samples, axis=0, ddof=0) / np.array([0.08, 0.06, 0.1])))
    return float(np.clip(np.exp(-penalty), 0.0, 1.0))

def panel_2e(states):
    columns = {}
    for label, frame in states.items():
        ordered = frame.sort_values('time_s', kind='mergesort')
        energy = ordered['energy'].to_numpy(float)
        columns[f'{label}_u'] = normalized_order(len(energy))
        columns[f'{label}_cumulative_AE_energy'] = np.cumsum(energy) / energy.sum()
    selected = states['91.15%'].sort_values('time_s', kind='mergesort')
    energy = selected['energy'].to_numpy(float)
    point, fitted, scan = segmented_fit(energy)
    rng = np.random.default_rng(20260920)
    points = []
    for _ in range(60):
        index = np.sort(rng.choice(len(energy), size=max(200, int(round(0.8 * len(energy)))), replace=False))
        points.append(segmented_fit(energy[index])[0])
    columns['91.15%_piecewise_fit'] = fitted
    columns['candidate_u'] = scan['candidate_u'].to_numpy()
    columns['BIC'] = scan['BIC'].to_numpy()
    columns['Delta_BIC'] = scan['Delta_BIC'].to_numpy()
    columns['log10_1_plus_Delta_BIC'] = scan['log10_1_plus_Delta_BIC'].to_numpy()
    columns['selected_change_point'] = np.array([point])
    columns['change_point_CI95_low'] = np.array([np.quantile(points, 0.025)])
    columns['change_point_CI95_high'] = np.array([np.quantile(points, 0.975)])
    columns['D_early'] = np.array([descriptor(selected)[1]])
    length = max(map(len, columns.values()))
    return pd.DataFrame({key: pd.Series(value, index=range(len(value))).reindex(range(length)) for key, value in columns.items()})

def panel_2f(states):
    point, _, delta, pre, post = descriptor(states['91.15%'])
    length = max(len(pre), len(post))
    result = pd.DataFrame({'Pre_RA': pd.Series(pre), 'Post_RA': pd.Series(post)}).reindex(range(length))
    result.loc[0, ['change_point_u', 'Delta_P_RA', 'P_post_gt_pre', 'n_pre', 'n_post']] = [point, delta, (delta + 1.0) / 2.0, len(pre), len(post)]
    return result

def panel_2g(states):
    rows = []
    for label, frame in states.items():
        values = np.sort(frame['amplitude_dB'].to_numpy(float))
        values = values[np.isfinite(values) & (values >= 50.0)]
        unique, counts = np.unique(values, return_counts=True)
        survival = np.cumsum(counts[::-1])[::-1]
        rows.extend(({'Filter_state': label, 'Amplitude_dB': x, 'log10_N_ge_A': np.log10(y)} for x, y in zip(unique, survival)))
    return pd.DataFrame(rows)

def panel_2h(states):
    thresholds = np.arange(45.0, 71.0, 1.0)
    streams = iter(np.random.SeedSequence(20260825).spawn(len(STATE_ORDER) * len(thresholds)))
    values = {label: states[label]['amplitude_dB'].to_numpy(float) for label in STATE_ORDER}
    results = {threshold: {} for threshold in thresholds}
    for label in STATE_ORDER:
        for threshold in thresholds:
            rng = np.random.default_rng(next(streams))
            results[threshold][label] = geometric_bootstrap_gof(values[label], threshold, rng, n_bootstrap=2000)
    rows = []
    for threshold in thresholds:
        state_results = results[threshold]
        probabilities = {label: state_results[label][4] for label in STATE_ORDER}
        counts = {label: state_results[label][3] for label in STATE_ORDER}
        minimum_p_state = min(STATE_ORDER, key=probabilities.get)
        minimum_tail_state = min(STATE_ORDER, key=counts.get)
        row = {
            'Completeness_threshold_dB': threshold,
            'Minimum_bootstrap_GOF_p': probabilities[minimum_p_state],
            'Minimum_p_state': minimum_p_state,
            'Minimum_tail_events': counts[minimum_tail_state],
            'Minimum_tail_state': minimum_tail_state,
            'GOF_probability_cutoff': 0.10,
            'Minimum_tail_event_requirement': 100,
        }
        for label in STATE_ORDER:
            q_value, b_value, distance, count, probability = state_results[label]
            row[f'{label}_q_MLE'] = q_value
            row[f'{label}_Amplitude_b_MLE'] = b_value
            row[f'{label}_GOF_distance'] = distance
            row[f'{label}_Tail_events'] = count
            row[f'{label}_bootstrap_GOF_p'] = probability
        row['Acceptable_common_threshold'] = probabilities[minimum_p_state] >= 0.10 and counts[minimum_tail_state] >= 100
        rows.append(row)
    result = pd.DataFrame(rows)
    acceptable = result.loc[result['Acceptable_common_threshold'], 'Completeness_threshold_dB']
    selected = float(acceptable.iloc[0]) if not acceptable.empty else np.nan
    result['Selected_common_threshold_dB'] = selected
    result['Is_selected_threshold'] = result['Completeness_threshold_dB'].eq(selected)
    return result

def panel_2i(states):
    threshold = 65.0
    rows = []
    for state_index, label in enumerate(STATE_ORDER):
        amplitude = states[label]['amplitude_dB'].to_numpy(float)
        q_value, b_value, distance, count = geometric_amplitude_fit(amplitude, threshold)
        tail = amplitude[np.isfinite(amplitude) & (amplitude >= threshold)]
        gaps = np.rint(tail - threshold).astype(int)
        rng = np.random.default_rng(20260828 + state_index)
        bootstrap_gaps = rng.choice(gaps, size=(2000, count), replace=True)
        bootstrap_mean = bootstrap_gaps.mean(axis=1)
        bootstrap_q = bootstrap_mean / (1.0 + bootstrap_mean)
        bootstrap_b = -20.0 * np.log10(bootstrap_q)
        low, high = np.quantile(bootstrap_b, [0.025, 0.975])
        for replicate, value in enumerate(bootstrap_b, start=1):
            rows.append({
                'Filter_state': label,
                'Completeness_threshold_dB': threshold,
                'Bootstrap_replicate': replicate,
                'Bootstrap_amplitude_b': value,
                'Amplitude_b_MLE': b_value,
                'CI95_low': low,
                'CI95_high': high,
                'q_MLE': q_value,
                'GOF_distance': distance,
                'Tail_events': count,
            })
    return pd.DataFrame(rows)

def panel_2j(states):
    rows = []
    for label in STATE_ORDER:
        point, d_early, delta, pre, post = descriptor(states[label])
        rows.append({'Filter_state': label, 'Change_point_u': point, 'D_early': d_early, 'Delta_P_RA': delta, 'Pre_window_events': len(pre), 'Post_window_events': len(post)})
    return pd.DataFrame(rows)

def panel_2k(data_path):
    counts = pd.read_excel(data_path, sheet_name='PU_State_Counts')
    samples = pd.read_excel(data_path, sheet_name='PU_Descriptor_Subsamples')
    counts['Filter_state'] = counts['Filter_state'].replace({'Raw': 'RAW'})
    raw_candidates = float(counts.loc[counts['Filter_state'].eq('RAW'), 'AE_candidate_events'].iloc[0])
    rows = []
    for label in STATE_ORDER:
        count = counts.loc[counts['Filter_state'].eq(label)].iloc[0]
        block = samples.loc[samples['Filter_state'].eq('Raw' if label == 'RAW' else label)]
        descriptor_values = block[['Change_point_u', 'D_early', 'Delta_P_RA']].to_numpy(float)
        purity = float(count['AE_candidate_events'] / count['Retained_events'])
        retention = float(count['AE_candidate_events'] / raw_candidates)
        stable = stability(descriptor_values)
        q_filter = float((purity * retention * stable) ** (1.0 / 3.0))
        sd = np.std(descriptor_values, axis=0, ddof=0)
        rows.append({'Filter_state': label, 'Removed_events_pct': count['Removed_events_pct'], 'Retained_events': int(count['Retained_events']), 'Qualified_events': int(count['AE_candidate_events']), 'Signal_purity': purity, 'Signal_retention': retention, 'Descriptor_stability': stable, 'Q_filter': q_filter, 'SD_u_star': sd[0], 'SD_D_early': sd[1], 'SD_Delta_P_RA': sd[2]})
    return pd.DataFrame(rows)

def calculate_panel(panel, data_path):
    if panel == '2k':
        return panel_2k(data_path)
    states = load_states(data_path)
    functions = {'2e': panel_2e, '2f': panel_2f, '2g': panel_2g, '2h': panel_2h, '2i': panel_2i, '2j': panel_2j}
    return functions[panel](states)

def main():
    parser = argparse.ArgumentParser(description='Export a main-text AE filtering and damage-descriptor calculation table.')
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Input workbook (default: repository data/Supplementary_Data.xlsx).')
    parser.add_argument('--panel', choices=['2e', '2f', '2g', '2h', '2i', '2j', '2k'], required=True)
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
