"""Export manuscript tables directly from the nine completed, paired runs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path('outputs_rate_adaptation')
OUT = Path(__file__).parent / 'paper_tables'
OUT.mkdir(exist_ok=True)
frames = {k: [] for k in ['metrics', 'per_class', 'per_subject', 'training', 'training_summary', 'initialization_audit']}
manifest = []
for pattern, split in [('bottleneck_split7*_20261003_234338', 7), ('bottleneck_split17*_20261003_234338', 17), ('confirmation_split37*_20261004_001812', 37)]:
    runs = sorted(ROOT.glob(pattern))
    assert len(runs) == 3
    for run in runs:
        seed = json.loads((run / 'protocol.json').read_text())['arguments']['seed']
        for key in frames:
            p = run / (key + '.csv')
            frames[key].append(pd.read_csv(p).assign(split=split, seed=seed))
            manifest.append({'source': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
data = {k: pd.concat(v, ignore_index=True) for k, v in frames.items()}
for key, frame in data.items():
    frame.to_csv(OUT / ('all_' + key + '.csv'), index=False)
(OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
names = {'blind_adapter': 'Early CE', 'early_distill': 'Early KD', 'late_adapter': 'Late CE', 'late_distill': 'Late KD', 'late_reverse_distill': 'Reverse KD'}
order = list(names)
m = data['metrics']

def write_table(name, header, rows, columns):
    text = '\\begin{tabular}{' + columns + '}\n\\toprule\n' + ' & '.join(header) + ' \\\\\n\\midrule\n'
    text += ''.join(' & '.join(map(str, row)) + ' \\\\\n' for row in rows)
    text += '\\bottomrule\n\\end{tabular}\n'
    (OUT / (name + '.tex')).write_text(text)

protocol = [
    ('Recordings', 'HARTH participants', '22'),
    ('Inputs', 'Four-second window', '120 steps, 6 channels'),
    ('Split 7 and 17', 'Train, validation, test windows', '2800, 800, 800'),
    ('Confirmation', 'Train, validation, test windows', '2800, 800, 3452'),
    ('Labels', 'Test-supported labels in 7, 17, 37', '11, 9, 12'),
    ('Acquisition', 'Training rates', '30, 15, 10 Hz'),
    ('Acquisition', 'Unseen test rates', '12, 6 Hz'),
    ('Backbone', 'Width, layers, patch length', '48, 8, 12'),
    ('Optimization', 'Updates, paired batch, seeds', '600, 32 pairs, 3'),
    ('Trainable', 'Head-only control', '8076'),
    ('Trainable', 'Early or late adapter plus head', '9948'),
    ('Trainable', 'Last-block control', '23205'),
]
pd.DataFrame(protocol, columns=['block', 'quantity', 'value']).to_csv(OUT / 'table1_protocol.csv', index=False)
write_table('table1_protocol', ['Block', 'Quantity', 'Value'], protocol, 'lll')

perf = []
tex = []
for condition, label in [('clean30', '30'), ('unseen12', '12'), ('unseen6', '6')]:
    for method in order:
        q = m[(m.split == 37) & (m.condition == condition) & (m.method == method)]
        row = {'rate_hz': int(label), 'method': method, 'n_per_seed': int(q.n.iloc[0]), 'seeds': len(q)}
        cells = [label, names[method]]
        for metric in ['accuracy', 'macro_f1', 'nll']:
            row[metric + '_mean'] = q[metric].mean()
            row[metric + '_sd'] = q[metric].std(ddof=1)
            cells.append('$%.3f\\pm%.3f$' % (q[metric].mean(), q[metric].std(ddof=1)))
        perf.append(row)
        tex.append(cells)
pd.DataFrame(perf).to_csv(OUT / 'table2_confirmation.csv', index=False)
write_table('table2_confirmation', ['Hz', 'Method', 'Accuracy', 'Macro-F1', 'NLL'], tex, 'llrrr')

p = m.pivot(index=['split', 'condition', 'seed'], columns='method', values='macro_f1')
contrasts = []
tex = []
for split in [7, 17, 37]:
    for condition, rate in [('clean30', 30), ('unseen12', 12), ('unseen6', 6)]:
        q = p.loc[(split, condition)]
        vals = [q.late_adapter-q.blind_adapter, q.early_distill-q.blind_adapter, q.late_distill-q.late_adapter]
        vals.append(vals[2]-vals[1])
        row = {'split': split, 'rate_hz': rate, 'interaction_positive_seeds': int((vals[3] > 0).sum())}
        for key, v in zip(['late_location', 'early_kd', 'late_kd', 'interaction'], vals):
            row[key + '_mean_pp'] = 100*v.mean()
            row[key + '_sd_pp'] = 100*v.std(ddof=1)
        contrasts.append(row)
        tex.append([str(split), str(rate)] + ['$%+.2f$' % (v.mean()*100) for v in vals] + [str(int((vals[3]>0).sum()))+'/3'])
pd.DataFrame(contrasts).to_csv(OUT / 'table3_interaction.csv', index=False)
write_table('table3_interaction', ['Split', 'Hz', 'Location', 'Early KD', 'Late KD', 'Interaction', 'Positive'], tex, 'lrrrrrl')

cl = data['per_class']
robust = []
tex = []
for threshold in [1, 10, 20, 50]:
    q = cl[(cl.split==37) & (cl.condition=='unseen6') & (cl.support>=threshold)]
    v = q.groupby(['seed','method']).f1.mean().unstack()
    gain = v.late_distill-v.blind_adapter
    count = q[q.method=='blind_adapter'].groupby('seed').size().iloc[0]
    row = {'audit':'minimum_class_support', 'group': str(threshold), 'count': int(count), 'gain_mean_pp':gain.mean()*100, 'gain_sd_pp':gain.std(ddof=1)*100, 'positive_seeds':int((gain>0).sum())}
    robust.append(row)
    tex.append(['Support $\\geq %d$'%threshold, str(count)+' labels', '$%+.2f\\pm%.2f$'%(gain.mean()*100,gain.std(ddof=1)*100),str(int((gain>0).sum()))+'/3'])
for subject in ['S015', 'S023', 'S025', 'S029']:
    q = data['per_subject']
    q = q[(q.split==37)&(q.condition=='unseen6')&(q.subject==subject)]
    v = q.pivot(index='seed', columns='method', values='macro_f1')
    gain = v.late_distill-v.blind_adapter
    n = int(q.n.iloc[0])
    robust.append({'audit':'subject_fixed12_f1','group':subject,'count':n,'gain_mean_pp':gain.mean()*100,'gain_sd_pp':gain.std(ddof=1)*100,'positive_seeds':int((gain>0).sum())})
    tex.append([subject, str(n)+' windows', '$%+.2f\\pm%.2f$'%(gain.mean()*100,gain.std(ddof=1)*100),str(int((gain>0).sum()))+'/3'])
pd.DataFrame(robust).to_csv(OUT / 'table4_robustness.csv', index=False)
write_table('table4_robustness', ['Audit', 'Support', 'F1 gain (pp)', 'Positive'], tex, 'llrl')
print(pd.DataFrame(contrasts).to_string(index=False))
print('Exported four tables and six complete source aggregates to', OUT)

final_frames = []
protocol_rows = []
for pattern, split in [('bottleneck_split7*_20261003_234338', 7), ('bottleneck_split17*_20261003_234338', 17), ('confirmation_split37*_20261004_001812', 37)]:
    for run in sorted(ROOT.glob(pattern)):
        protocol = json.loads((run / 'protocol.json').read_text())
        seed = protocol['arguments']['seed']
        source = run / 'final_metrics.csv'
        final_frames.append(pd.read_csv(source).assign(split=split, seed=seed))
        manifest.append({'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        for part, counts in protocol['class_counts'].items():
            for label, count in counts.items():
                protocol_rows.append({'split': split, 'seed': seed, 'partition': part, 'class': int(label), 'support': count})
final = pd.concat(final_frames, ignore_index=True)
final.to_csv(OUT / 'supplement_final_metrics.csv', index=False)
fp = final.pivot(index=['split', 'condition', 'seed'], columns='method', values='macro_f1')
fp['interaction'] = (fp.late_distill-fp.late_adapter)-(fp.early_distill-fp.blind_adapter)
fp['late_kd_minus_early_ce'] = fp.late_distill-fp.blind_adapter
fp[['interaction', 'late_kd_minus_early_ce']].to_csv(OUT / 'supplement_final_contrasts.csv')
pd.DataFrame(protocol_rows).to_csv(OUT / 'supplement_class_support.csv', index=False)
m[m.condition.isin(['aa15', 'aa10', 'naive15', 'naive10'])].to_csv(OUT / 'supplement_rate_controls.csv', index=False)
(OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
print('Exported fixed-step, acquisition-control, and class-support supplements')
