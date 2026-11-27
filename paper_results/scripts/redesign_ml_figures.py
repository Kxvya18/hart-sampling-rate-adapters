"""Replace figures 1 and 3 only; preserve the accepted figures 2 and 4."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

ROOT=Path('outputs_rate_adaptation')
OUT=Path(__file__).parent/'ml_figures'
parts=[];classes=[]
for pattern,split in [('bottleneck_split7*_20261003_234338',7),('bottleneck_split17*_20261003_234338',17),('confirmation_split37*_20261004_001812',37)]:
    for run in sorted(ROOT.glob(pattern)):
        seed=int(run.name.split('_')[2][4:])
        parts.append(pd.read_csv(run/'predictions.csv').assign(split=split,seed=seed))
        classes.append(pd.read_csv(run/'per_class.csv').assign(split=split,seed=seed))
p=pd.concat(parts,ignore_index=True);c=pd.concat(classes,ignore_index=True)
p['nll']=-np.log(p.p_true.clip(1e-12,1))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,
                     'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42})
grey=LinearSegmentedColormap.from_list('dense_grey',['.8','.02'])
def canvas():
    f,a=plt.subplots(2,2,figsize=(9.4,7.3),layout='constrained')
    f.set_constrained_layout_pads(w_pad=.12,h_pad=.12,wspace=.10,hspace=.12)
    return f,a.ravel()
def legend(a,cols=2):
    a.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=cols,frameon=False)
    count=len(a.get_legend_handles_labels()[0])
    rows=(count+cols-1)//cols
    a.set_title(a.get_title(),pad=26 if rows==1 else 49)
def save(f,name):
    f.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    f.savefig(OUT/(name+'.png'),dpi=280,bbox_inches='tight');plt.close(f)
def windows(split,condition,value='nll'):
    return p[(p.split==split)&(p.condition==condition)].groupby(['subject','window_index','true','method'])[value].mean().unstack()

f,a=canvas();q=windows(37,'unseen6','p_true')
im=a[0].hexbin(q.early_distill,q.late_distill,gridsize=42,extent=(0,1,0,1),mincnt=1,bins='log',cmap=grey,linewidths=0)
a[0].plot([0,1],[0,1],':',color='.5')
a[0].set(xlim=(0,1),ylim=(0,1),xlabel='Early-KD true-label probability',ylabel='Late-KD true-label probability',title='(a) Location changes task evidence')
f.colorbar(im,ax=a[0],label='Observed-window count (log)',shrink=.85)
for split,color in [(7,'.65'),(17,'.35'),(37,'0')]:
    q=windows(split,'unseen6')
    for location,base,kd,style in [('Early','blind_adapter','early_distill','--'),('Late','late_adapter','late_distill','-')]:
        benefit=np.sort((q[base]-q[kd]).values)
        a[1].plot(benefit,(np.arange(len(benefit))+.5)/len(benefit),color=color,ls=style,lw=1.5,label=f'{location}, {split}')
a[1].axvline(0,color='.7',ls=':')
a[1].set(xlim=(-1.5,1.5),ylim=(0,1),xlabel='CE minus KD per-window NLL',ylabel='Empirical cumulative fraction',title='(b) Where distillation helps or harms')
legend(a[1],3)
q=windows(37,'unseen6')
order=q.blind_adapter.rank(method='first',pct=True)
bins=np.minimum((order*20).astype(int),19)
for method,color,style,label in [('early_distill','.65','--','Early KD'),('late_adapter','.35','-.','Late CE'),('late_distill','0','-','Late KD'),('late_reverse_distill','.5',':','Late reverse KD')]:
    values=(q.blind_adapter-q[method]).groupby(bins).mean()
    a[2].plot((values.index+.5)/20,values.values,color=color,ls=style,marker='o',ms=3,label=label)
a[2].axhline(0,color='.7',ls=':')
a[2].set(xlabel='Early-CE loss quantile (easy → difficult)',ylabel='Mean NLL reduction versus early CE',title='(c) Gains across baseline difficulty')
legend(a[2],2)
interaction=(q.late_adapter-q.late_distill)-(q.blind_adapter-q.early_distill)
im=a[3].hexbin(q.blind_adapter,interaction,gridsize=40,extent=(0,6,-1,1),mincnt=1,bins='log',cmap=grey,linewidths=0)
a[3].axhline(0,color='.5',ls=':')
a[3].set(xlim=(0,6),ylim=(-1,1),xlabel='Early-CE per-window mean NLL',ylabel='Location × KD NLL interaction',title='(d) Paired objective interaction density')
f.colorbar(im,ax=a[3],label='Observed-window count (log)',shrink=.85)
save(f,'figure1_location_objective')

f,a=canvas();labels=[1,2,3,4,5,6,7,8,13,14,130,140]
matrix=[];rowlabels=[]
for split in [7,17,37]:
    for seed in [7,17,27]:
        frame=c[(c.split==split)&(c.seed==seed)&(c.condition=='unseen6')]
        pivot=frame.pivot(index='class',columns='method',values='f1')
        support=frame.groupby('class').support.max()
        matrix.append((pivot.late_distill-pivot.blind_adapter).where(support>0).reindex(labels).values)
        rowlabels.append(f'{"C" if split==37 else split} · {seed}')
im=a[0].imshow(matrix,cmap='Greys',norm=TwoSlopeNorm(0,-.4,.4),aspect='auto')
a[0].set(xticks=range(12),xticklabels=labels,yticks=range(9),yticklabels=rowlabels,xlabel='Activity ID',ylabel='Partition · model seed',title='(a) Class effects across every run')
a[0].tick_params(axis='x',rotation=45)
f.colorbar(im,ax=a[0],label='Late KD minus early CE F1',shrink=.85)
q=windows(37,'unseen6');benefit=q.blind_adapter-q.late_distill
arrays=[benefit[q.index.get_level_values('true')==label].values for label in labels]
violin=a[1].violinplot(arrays,positions=range(12),widths=.8,showmeans=False,showmedians=True,showextrema=False)
for body in violin['bodies']:
    body.set_facecolor('.45');body.set_edgecolor('.1');body.set_alpha(.7)
violin['cmedians'].set_color('0')
a[1].axhline(0,color='.6',ls=':')
a[1].set(xticks=range(12),xticklabels=labels,ylim=(-3,3),xlabel='Activity ID',ylabel='Early CE minus late KD NLL',title='(b) Within-class loss distributions')
a[1].tick_params(axis='x',rotation=45)
thresholds=[1,5,10,15,20,25,30,50,100]
for split,color in [(7,'.65'),(17,'.35'),(37,'0')]:
    traces=[]
    for seed in [7,17,27]:
        frame=c[(c.split==split)&(c.seed==seed)&(c.condition=='unseen6')]
        values=[]
        for threshold in thresholds:
            averages=frame[frame.support>=threshold].groupby('method').f1.mean()
            values.append((averages.late_distill-averages.blind_adapter)*100)
        traces.append(values)
        a[2].plot(range(len(thresholds)),values,color=color,lw=.65,alpha=.25)
    traces=np.array(traces)
    a[2].plot(range(len(thresholds)),traces.mean(0),color=color,lw=1.7,marker='o',ms=3,label=f'{"Confirmation" if split==37 else "Split "+str(split)}')
    a[2].fill_between(range(len(thresholds)),traces.min(0),traces.max(0),color=color,alpha=.09)
a[2].axhline(0,color='.6',ls=':')
a[2].set(xticks=range(len(thresholds)),xticklabels=thresholds,xlabel='Minimum windows per averaged class',ylabel='Macro-F1 gain (pp)',title='(c) Support sensitivity across partitions')
legend(a[2],3)
for i,subject in enumerate(sorted(q.index.get_level_values('subject').unique())):
    values=np.sort(benefit[q.index.get_level_values('subject')==subject].values)
    a[3].plot(values,(np.arange(len(values))+.5)/len(values),color=str(.65-.2*i),ls=[':', '--','-.','-'][i],lw=1.6,label=subject)
a[3].axvline(0,color='.6',ls=':')
a[3].set(xlim=(-2,3),ylim=(0,1),xlabel='Early CE minus late KD NLL',ylabel='Empirical cumulative fraction',title='(d) Participant-specific loss changes')
legend(a[3],2)
save(f,'figure3_class_subject_audit')
(OUT/'REDESIGN_NOTES.md').write_text('Figures 1 and 3 were redesigned. Figures 2 and 4 are unchanged. Each observed window is averaged across model seeds before probability, NLL-density, violin or participant CDF display. Thus repeated model predictions are not counted as new recordings. Split labels 7 and 17 are exploratory and C denotes confirmation 37. Positive CE-minus-KD NLL indicates KD improvement. The NLL location interaction is (late CE minus late KD) minus (early CE minus early KD), distinct from the F1 interaction in the earlier figure version. Loss-quantile comparisons are retrospective descriptive analyses, not model features. Violin shapes are kernel density estimates of actual window losses, not synthetic samples. Display ranges clip axes, not the underlying metrics. Shading spans seeds, not population confidence intervals. Class thresholds change the averaged class set while retaining all test predictions. Class heatmap colours saturate at ±0.4 and absent classes are blank.\n')
print('Replaced figures 1 and 3 only')
