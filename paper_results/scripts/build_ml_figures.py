"""Quantitative monochrome panels from saved HART experiments only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

ROOT = Path('outputs_rate_adaptation')
OUT = Path(__file__).parent / 'ml_figures'
OUT.mkdir(exist_ok=True)
METHODS = ['blind_adapter','early_distill','late_adapter','late_distill','late_reverse_distill']
NAMES = ['Early CE','Early KD','Late CE','Late KD','Late reverse KD']
COLORS = ['.65','.45','.25','0','.8']
STYLES = [':','--','-.','-','--']
LABELS = [1,2,3,4,5,6,7,8,13,14,130,140]
frames = {k:[] for k in ['metrics','predictions','per_class','per_subject','training']}
provenance = []
for pattern, split in [('bottleneck_split7*_20261003_234338',7),('bottleneck_split17*_20261003_234338',17),('confirmation_split37*_20261004_001812',37)]:
    runs = sorted(ROOT.glob(pattern))
    assert len(runs) == 3
    for run in runs:
        seed = json.loads((run/'protocol.json').read_text())['arguments']['seed']
        for key in frames:
            path = run/(key+'.csv')
            frames[key].append(pd.read_csv(path).assign(split=split,seed=seed))
            provenance.append({'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
data = {k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
m = data['metrics']; pred = data['predictions']; cl = data['per_class']
confirm = m[m.split==37]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,
                     'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})

def canvas():
    fig, axes = plt.subplots(2,2,figsize=(9.4,7.3),layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.12,h_pad=.12,wspace=.08,hspace=.10)
    return fig,axes.ravel()

def legend(ax,columns=2):
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=columns,frameon=False,
              borderaxespad=0,handlelength=2.5,columnspacing=.9)

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=280,bbox_inches='tight')
    plt.close(fig)

def line_summary(ax,frame,metric,condition_order=['unseen6','aa10','unseen12','aa15','clean30']):
    for i,method in enumerate(METHODS):
        s=frame[frame.method==method].groupby('condition')[metric].agg(['mean','min','max']).reindex(condition_order)
        x=np.arange(len(s))
        ax.plot(x,s['mean'],color=COLORS[i],ls=STYLES[i],marker=['o','s','^','D','v'][i],ms=3.5,label=NAMES[i])
        ax.fill_between(x,s['min'],s['max'],color=COLORS[i],alpha=.08)
    ax.set(xticks=np.arange(5),xticklabels=['6','10','12','15','30'],xlabel='Acquisition rate (Hz)',ylabel=metric.replace('_',' ').title())
    legend(ax)
    ax.set_title(ax.get_title(),pad=49)

pivot=m.pivot(index=['split','seed','condition'],columns='method',values='macro_f1')
interaction=(pivot.late_distill-pivot.late_adapter)-(pivot.early_distill-pivot.blind_adapter)
interaction.rename('f1_interaction').to_csv(OUT/'interaction.csv')
fig,ax=canvas()
ax[0].set_title('(a) Confirmation rate transfer')
line_summary(ax[0],confirm,'macro_f1')
for i,split in enumerate([7,17,37]):
    vals=interaction.loc[(split,slice(None),'unseen6')].values*100
    # Ordered seed positions, not simulated points or random jitter.
    positions=i+np.array([-.16,0,.16])
    ax[1].plot(positions,vals,'o',color=str(.55-.22*i),ms=5)
    ax[1].hlines(vals.mean(),i-.23,i+.23,color='0',lw=2)
ax[1].axhline(0,color='.6',ls=':',lw=1)
ax[1].set(xticks=[0,1,2],xticklabels=['Split 7','Split 17','Confirmation'],ylabel='Location × KD interaction (F1 pp)',title='(b) Paired 6 Hz interaction')
for i,method in enumerate(['blind_adapter','late_adapter']):
    target='early_distill' if method=='blind_adapter' else 'late_distill'
    for j,split in enumerate([7,17,37]):
        q=pivot.loc[(split,slice(None),'unseen6')]
        ax[2].scatter(q[method],q[target],marker=['o','s','^'][j],s=35,facecolor=['.7','0'][i],edgecolor='0',label=f'{["Early","Late"][i]}, {split}')
lo,hi=.22,.53
ax[2].plot([lo,hi],[lo,hi],color='.6',ls=':')
ax[2].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Cross-entropy macro-F1',ylabel='With distillation macro-F1',title='(c) Objective effect at fixed location')
legend(ax[2],3);ax[2].set_title(ax[2].get_title(),pad=38)
matrix=np.array([[interaction.loc[(s,slice(None),c)].mean()*100 for c in ['clean30','unseen12','unseen6']] for s in [7,17,37]])
lim=max(abs(matrix).max(),1)
im=ax[3].imshow(matrix,cmap='Greys',norm=TwoSlopeNorm(0,-lim,lim),aspect='auto')
ax[3].set(xticks=[0,1,2],xticklabels=['30 Hz','12 Hz','6 Hz'],yticks=[0,1,2],yticklabels=['Split 7','Split 17','Confirm'],title='(d) Interaction across acquisition regimes')
for y in range(3):
    for x in range(3): ax[3].text(x,y,f'{matrix[y,x]:+.2f}',ha='center',va='center',color='white' if matrix[y,x]>lim*.45 else 'black',fontsize=9)
fig.colorbar(im,ax=ax[3],label='Mean interaction (F1 pp)',shrink=.85)
save(fig,'figure1_location_objective')

cp=pred[pred.split==37].copy()
cp['nll']=-np.log(cp.p_true.clip(1e-12,1))
fig,ax=canvas()
for axis,condition,title in [(ax[0],'unseen6','(a) Severe-shift loss tails'),(ax[1],'unseen12','(b) Intermediate-shift loss tails')]:
    for i,method in enumerate(METHODS):
        # Average model-seed loss within each observed window before displaying distributions.
        values=cp[(cp.method==method)&(cp.condition==condition)].groupby(['subject','window_index']).nll.mean().sort_values().values
        tail=(len(values)-np.arange(len(values)))/len(values)
        axis.plot(values,tail,color=COLORS[i],ls=STYLES[i],label=NAMES[i],lw=1.4)
    axis.set(xlim=(0,8),ylim=(1e-3,1),yscale='log',xlabel='Per-window mean NLL',ylabel='Empirical tail fraction',title=title)
    legend(axis);axis.set_title(title,pad=49)
q=cp[cp.condition=='unseen6'].groupby(['subject','window_index','method']).nll.mean().unstack()
im=ax[2].hexbin(q.blind_adapter,q.late_distill,gridsize=38,extent=(0,8,0,8),mincnt=1,cmap=LinearSegmentedColormap.from_list('visible_greys',['.8','.05']),bins='log',linewidths=0)
ax[2].plot([0,8],[0,8],ls=':',color='.5')
ax[2].set(xlim=(0,8),ylim=(0,8),xlabel='Early-adapter mean NLL',ylabel='Late-distillation mean NLL',title='(c) Paired observed-window losses')
fig.colorbar(im,ax=ax[2],label='Window count (log scale)',shrink=.85)
for i,seed in enumerate([7,17,27]):
    q=cp[(cp.condition=='unseen6')&(cp.seed==seed)].pivot(index=['subject','window_index'],columns='method',values='nll')
    gain=q.blind_adapter-q.late_distill
    bins=np.linspace(-4,4,65)
    ax[3].hist(gain,bins=bins,weights=np.full(len(gain),1/(len(gain)*(bins[1]-bins[0]))),histtype='step',lw=1.5,color=str(.65-.3*i),label=f'Seed {seed}')
ax[3].axvline(0,color='.6',ls=':');ax[3].set(xlabel='Early minus late-KD NLL',ylabel='Density',title='(d) Seedwise paired loss improvements',xlim=(-4,4))
legend(ax[3],3);ax[3].set_title(ax[3].get_title(),pad=24)
save(fig,'figure2_prediction_distributions')

fig,ax=canvas()
gains=[]
for split in [7,17,37]:
    q=cl[(cl.split==split)&(cl.condition=='unseen6')].groupby(['class','method']).f1.mean().unstack()
    support=cl[(cl.split==split)&(cl.condition=='unseen6')].groupby('class').support.max()
    gain=(q.late_distill-q.blind_adapter).where(support>0)
    gains.append(gain.reindex(LABELS).values)
gains=np.array(gains)
im=ax[0].imshow(gains,cmap='Greys',norm=TwoSlopeNorm(0,-.4,.4),aspect='auto')
ax[0].set(xticks=range(12),xticklabels=LABELS,yticks=range(3),yticklabels=['Split 7','Split 17','Confirm'],xlabel='Activity ID',title='(a) Class-wise 6 Hz F1 changes')
ax[0].tick_params(axis='x',rotation=45)
fig.colorbar(im,ax=ax[0],label='Late KD minus early CE',shrink=.85)
counts=cl[(cl.split==37)&(cl.seed==7)&(cl.condition=='unseen6')&(cl.method=='blind_adapter')].set_index('class').support.reindex(LABELS)
ax[1].bar(range(12),counts,color='.3',width=.7)
ax[1].set(xticks=range(12),xticklabels=LABELS,yscale='log',ylabel='Observed windows (log scale)',xlabel='Activity ID',title='(b) Confirmation class support');ax[1].tick_params(axis='x',rotation=45)
for i,seed in enumerate([7,17,27]):
    values=[]
    q=cl[(cl.split==37)&(cl.seed==seed)&(cl.condition=='unseen6')]
    for threshold in [1,10,20,50,100]:
        f=q[q.support>=threshold].groupby('method').f1.mean()
        values.append((f.late_distill-f.blind_adapter)*100)
    ax[2].plot(range(5),values,marker=['o','s','^'][i],color=str(.65-.3*i),label=f'Seed {seed}')
ax[2].axhline(0,color='.6',ls=':');ax[2].set(xticks=range(5),xticklabels=[1,10,20,50,100],xlabel='Minimum windows per averaged class',ylabel='Macro-F1 gain (pp)',title='(c) Support-threshold audit')
legend(ax[2],3);ax[2].set_title(ax[2].get_title(),pad=24)
s=data['per_subject'];s=s[(s.split==37)&(s.condition=='unseen6')]
q=s.pivot(index=['subject','seed'],columns='method',values='macro_f1')
for i,subject in enumerate(sorted(s.subject.unique())):
    for j,seed in enumerate([7,17,27]):
        delta=(q.loc[(subject,seed),'late_distill']-q.loc[(subject,seed),'blind_adapter'])*100
        ax[3].scatter(i+(j-1)*.13,delta,marker=['o','s','^'][j],color=str(.65-.3*j),s=32,label=f'Seed {seed}' if i==0 else None)
    mean=(q.loc[subject].late_distill-q.loc[subject].blind_adapter).mean()*100
    ax[3].hlines(mean,i-.22,i+.22,color='0',lw=2)
ax[3].axhline(0,color='.6',ls=':');ax[3].set(xticks=range(4),xticklabels=sorted(s.subject.unique()),ylabel='Per-subject F1 gain (pp)',title='(d) Subject-level effects')
legend(ax[3],3);ax[3].set_title(ax[3].get_title(),pad=24)
save(fig,'figure3_class_subject_audit')

fig,ax=canvas()
t=data['training'];t=t[t.split==37]
for i,method in enumerate(METHODS):
    q=t[t.method==method].groupby('epoch').validation_nll.agg(['mean','min','max'])
    ax[0].plot(q.index,q['mean'],color=COLORS[i],ls=STYLES[i],label=NAMES[i],lw=1.4)
    ax[0].fill_between(q.index,q['min'],q['max'],color=COLORS[i],alpha=.07)
ax[0].set(xlabel='Epoch',ylabel='Validation NLL',title='(a) Validation learning dynamics');legend(ax[0]);ax[0].set_title(ax[0].get_title(),pad=49)
for i,method in enumerate(['early_distill','late_distill','late_reverse_distill']):
    q=t[t.method==method].groupby('epoch').distillation.agg(['mean','min','max'])
    ax[1].plot(q.index,q['mean'],color=['.6','0','.3'][i],ls=['--','-','-.'][i],label=['Early KD','Late KD','Reverse KD'][i])
    ax[1].fill_between(q.index,q['min'],q['max'],color=['.6','0','.3'][i],alpha=.09)
ax[1].set(xlabel='Epoch',ylabel='Training temperature-scaled KL',title='(b) Paired-view objective dynamics');legend(ax[1]);ax[1].set_title(ax[1].get_title(),pad=38)
matrices=[]
for method in ['blind_adapter','late_distill']:
    q=cp[(cp.method==method)&(cp.condition=='unseen6')]
    counts=confusion_matrix(q.true,q.predicted,labels=LABELS)
    matrices.append(counts/np.maximum(counts.sum(1,keepdims=True),1))
delta=matrices[1]-matrices[0]
im=ax[2].imshow(delta,cmap='Greys',norm=TwoSlopeNorm(0,-.4,.4))
ax[2].set(xticks=range(12),xticklabels=LABELS,yticks=range(12),yticklabels=LABELS,xlabel='Predicted activity ID',ylabel='Observed activity ID',title='(c) Change in row-normalized confusion');ax[2].tick_params(axis='x',rotation=45)
fig.colorbar(im,ax=ax[2],label='Late KD minus early CE',shrink=.8)
for i,method in enumerate(METHODS):
    q=cp[(cp.method==method)&(cp.condition=='unseen6')].copy()
    q['bin']=np.minimum((q.confidence*10).astype(int),9)
    q['correct']=(q.true==q.predicted).astype(float)
    stat=q.groupby('bin').agg(confidence=('confidence','mean'),accuracy=('correct','mean'),n=('correct','size'))
    ax[3].plot(stat.confidence,stat.accuracy,color=COLORS[i],ls=STYLES[i],marker='o',ms=3,label=NAMES[i])
ax[3].plot([0,1],[0,1],color='.7',ls=':');ax[3].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted confidence',ylabel='Empirical accuracy',title='(d) Confidence versus correctness');legend(ax[3]);ax[3].set_title(ax[3].get_title(),pad=49)
save(fig,'figure4_learning_errors')
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
(OUT/'FIGURE_NOTES.md').write_text('Figures use saved observed predictions only. CE denotes cross-entropy and KD online paired-view distillation. Shading spans three model seeds, not a population confidence interval. Per-window loss panels average seed losses before displaying distributions. Confusion and confidence diagnostics pool repeated predictions descriptively; they are not independent recordings. Heatmap colour scales clip at plus or minus 0.4, while underlying values are unmodified. Class heatmaps leave absent classes blank. Confirmation includes all twelve labels. Tail panels display NLL up to eight, and paired-loss plots use the labelled display ranges without renormalizing away excluded observations. Each comparison is paired within seed. Four test participants remain the effective confirmation subject sample.\n')
print('Saved four monochrome four-panel figures in',OUT)
