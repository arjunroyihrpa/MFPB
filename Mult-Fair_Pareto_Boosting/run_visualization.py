# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:59:28 2020

@author: Arjun
"""
import matplotlib.pyplot as plt
import numpy as np

from Maximus_optimized_non_dominated import Multi_Fair as maximus
from sklearn.model_selection import train_test_split



def cos_vis(path='',cost=[],th=0,N=0,L=['Marital','Race','Sex','Age'],dt='KDD',clf='Multi-Fair'):
    costs=[]
    for r in cost:
        vals=r.split(',')
        costs.append([float(v)-1 for v in vals[:4*N]])
    costs=np.array(costs)
    pp,npp,pn,npn=[],[],[],[]
    for i in range(N):
        pp.append(costs[:,i*4])
        npp.append(costs[:,i*4+1])
        pn.append(costs[:,i*4+2])
        npn.append(costs[:,i*4+3])
    fig,ax= plt.subplots(2,2, figsize=[30,10]) 
    cnt=0
    for i in range(2):
        for j in range(2):
           if(cnt<N):
               tp=max([max(pp[cnt]),max(npp[cnt]),max(pn[cnt]),max(npn[cnt])])
               ax[i][j].set_ylim(bottom=0.0,top=0.4)
               ax[i][j].plot(pp[cnt], label='prot-pos '+L[cnt],linestyle='-')
               ax[i][j].plot(npp[cnt], label='nonprot-pos '+L[cnt],linestyle='-.')
               ax[i][j].plot(pn[cnt], label='prot-neg '+L[cnt],linestyle=':')
               ax[i][j].plot(npn[cnt], label='nonprot-neg '+L[cnt],linestyle='--')
               ax[i][j].axvline(x=th, color='k', linestyle='--',label='convergence')
               ax[i][j].legend(loc='upper right', fancybox=True, framealpha=0.1)
               ax[i][j].set_title('Costs '+L[cnt])
               cnt+=1
    fig.suptitle('Multi-protected Fairness costs over t',fontsize='large', fontweight='bold')
    font = {'family' : 'monospace',
        'weight' : 'bold',
        }
    plt.rc('font', **font,size=30)
    plt.draw()
    #plt.savefig(path+'costs_'+dt+'.png')
    plt.show()
    
    
def obj_vis(classifier):
    objectives=np.array(classifier.ob)
    th=classifier.theta
    fig,ax=plt.subplots(figsize=[30,10],tight_layout=True)
    font = {'family' : 'monospace',
        'weight' : 'bold',
        }
    ax.plot(objectives[:,0],label='O1',linestyle="-",linewidth=3)

    ax.plot(objectives[:,1],label='O2',linestyle="-.",linewidth=3)
    ax.plot(objectives[:,2],label='O3',linestyle="--",linewidth=3)
    ax.axvline(x=th, color='k', linestyle=':',label='optimal convergenge',linewidth=3)
    ax.legend(loc='upper right', fancybox=True, framealpha=0.2)
    plt.ylabel('Loss',fontsize='large', fontweight='bold')
    plt.xlabel('number of rounds (t)',fontsize='large', fontweight='bold')
    plt.rc('font', **font,size=30)
    plt.show()

def vis_attribute_domination(cost,N,sensitives,th):
    costs=[]
    for r in cost:
        vals=r.split(',')
        costs.append([float(v)-1 for v in vals[:4*N]])

    costs=np.array(costs)
    pp,npp,pn,npn=[],[],[],[]
    for i in range(N):
        pp.append(costs[:,i*4])
        npp.append(costs[:,i*4+1])
        pn.append(costs[:,i*4+2])
        npn.append(costs[:,i*4+3])
    fc=[]
    for i in range(N):
        cp=[max(pp[i][j],npp[i][j],pn[i][j],npn[i][j]) for j in range(len(pp[i]))]
        fc.append(cp)
    fig,ax=plt.subplots(figsize=[30,10])
    st=['-',':','-.','-.']
    for i in range(N):
        ax.plot(fc[i],label=sensitives[i],linestyle=st[i],linewidth=3)

    ax.axvline(th, color='k', linestyle=':',label='optimal convergenge',linewidth=3)
    ax.legend(loc='upper right', fancybox=True, framealpha=0.2)
    fig.suptitle('Plots of the maximum fairness costs w.r.t to each sensitives')
    plt.ylabel('Loss')
    plt.xlabel('number of rounds (t)')
    plt.rc('font', size=20)
    plt.show()


def weight_plots(classifier,protected):
    iw=classifier.weight_list[0]
    wg=classifier.weight_list[1:classifier.theta]
    wg=np.array([[float(v) for v in w.split(',')] for w in wg])
    wgs=[sum(wg[:,i])/len(wg) for i in range(len(wg[0]))]
    wg=wgs
    iw=[float(v) for v in iw.split(',')[1:]]
    wg=np.array(wg)
    
    iw=np.array(iw)
    
    w,g=3,4
    prot_pos,non_prot_pos,prot_neg,non_prot_neg=[],[],[],[]
    for i in range(len(protected)):
        prot_pos.append(iw[w+6*i])
        prot_pos.append(wg[g+6*i])
        
        non_prot_pos.append(iw[w+1+6*i])
        non_prot_pos.append(wg[g+1+6*i])
        
        prot_neg.append(iw[w+2+6*i])
        prot_neg.append(wg[g+2+6*i])
        
        non_prot_neg.append(iw[w+3+6*i])
        non_prot_neg.append(wg[g+3+6*i])
    index = np.arange(len(prot_pos))
    bar_width=0.2
    patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
    fig,  ax3 = plt.subplots(figsize=(10,10))
    #fig.suptitle("Weights init vs_Final Multi-Fair")
    #ax3.bar(index, Tot_pos,label='Total. Pos.', edgecolor='yellow', width= bar_width)
    #ax3.bar(index, prot_pos,label='Prot. Pos.', edgecolor='black', width= bar_width)
    ax3.bar(index, prot_pos,label='Prot. Pos.', edgecolor='black', width= bar_width,hatch=patterns[0])
    ax3.bar(index, non_prot_pos,label='Non-Prot. Pos.', bottom=prot_pos, edgecolor='red', width= bar_width,hatch=patterns[1])
    ax3.bar(index, prot_neg,label='Prot. Neg.', bottom=[i+j for i,j in zip(prot_pos, non_prot_pos)],  edgecolor='green', width= bar_width,hatch=patterns[2])
    ax3.bar(index, non_prot_neg,label='Non-Prot. Neg.', bottom=[i+j+z for i,j,z in zip(prot_pos, non_prot_pos, prot_neg)],  edgecolor='blue', width= bar_width,hatch=patterns[3])
    
    ax3.set_xticks([i for i in range(len(index))])
    ax3.grid(True)
    labels=[]
    for p in protected:
        labels.append(p+'_ini')
        labels.append(p)
        ax3.set_xticklabels(labels)
        ax3.legend(loc='best', fancybox=True, framealpha=0.1)
    #plt.yticks(numpy.arange(0, 1.0001, step=0.1))
    ax3.set_ylim(top=1.2)
    #ax3.set_ylim([0.48, 0.52])
    font = {'family' : 'monospace',
        'weight' : 'bold',
        }
    plt.rc('font', **font,size=30)
    #plt.rcParams.update({'font.size': 25})
    plt.draw()
    #plt.savefig(path+'costs_'+dt+'.png')
    plt.show()

def run_debug(v):
    if v=='Adult':
        from DataPreprocessing.load_adult import load_adult
        X, y, sa_index, p_Group, x_control,F = load_adult()
        
    elif v=='Bank':
        from DataPreprocessing.load_bank import load_bank
        X, y, sa_index, p_Group, x_control,F = load_bank()

    elif v=='Credit':
        from DataPreprocessing.load_credit import load_credit
        X, y, sa_index, p_Group, x_control,F = load_credit()

    elif v=='Compas':
        from DataPreprocessing.load_compas_data import load_compas
        X, y, sa_index, p_Group, x_control,F = load_compas()
        
    elif v=='KDD':
        from DataPreprocessing.load_kdd import load_kdd
        X, y, sa_index, p_Group, x_control,F = load_kdd()

    protected=[F[v] for v in sa_index]
    if v=='Compas':
        protected[0]='age'
        
    path='Multi-Fair_Pareto_Boosting'
    obj,fair,thetas=[],[],[]
    
    cf1 = maximus(n_estimators=499, saIndex=sa_index, saValue=p_Group,debug=True)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    cf1.fit(X_train,y_train)
    cost=cf1.costs_list
    th=cf1.theta    
    cos_vis(path,cost,th,len(protected),protected,dt=v,clf='Maximus')
    obj.append(cf1.final_objective)
    fair.append(cf1.fairloss)
    thetas.append(th)    
    obj_vis(cf1)
    weight_plots(cf1,protected)

if __name__ == "__main__":
    run_debug('Compas')
'''

for dt in ['Adult','Bank','Credit','Compass']:
    if dt=='Adult':
        from DataPreprocessing.load_adult import load_adult
        X, y, sa_index, p_Group, x_control,F = load_adult()
        #v='Adult_2_sensi_Mari_Sex'
        saf=sa_index[1]
    elif dt=='Bank':
        from DataPreprocessing.load_bank import load_bank
        X, y, sa_index, p_Group, x_control,F = load_bank()
        saf=sa_index[0]
        print(saf)
    elif dt=='Credit':
        from DataPreprocessing.load_credit import load_credit
        X, y, sa_index, p_Group, x_control,F = load_credit()
        saf=sa_index[0]
    elif dt=='Compas':
        from DataPreprocessing.load_compas_data import load_compas
        X, y, sa, p_G, x_control,F = load_compas()
        sa_index=[sa[-1],sa[0]]
        p_Group=[p_G[-1],p_G[0]]
    sensitives=[F[v] for v in sa_index]
    in_ts,pred1=train(X,y)
    results.append(list(get_fairness(sa_index,p_Group,in_ts,pred1,X,y).values()))
    performance.append(get_score(pred1,in_ts,X,y))
'''