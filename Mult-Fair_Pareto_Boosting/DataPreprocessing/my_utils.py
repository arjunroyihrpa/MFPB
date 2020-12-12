# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:37:19 2020

@author: Arjun
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 01:25:21 2020

@author: Arjun
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix,roc_auc_score
from imblearn.metrics import geometric_mean_score
def get_score(pred,in_ts,X,y):
  resg=[]
  for j in range(len(pred)):        
    ts,ytest=in_ts[j],[]
    TP,TN,FP,FN=[],[],[],[]
    for i in range(len(ts)):
        ytest.append(y[ts[i]])
        if y[ts[i]]==1 and pred[j][i]==1:
             TP.append(ts[i])
        elif y[ts[i]]==-1 and pred[j][i]==-1:
             TN.append(ts[i])
        elif y[ts[i]]==1 and pred[j][i]==-1:
             FN.append(ts[i])
        elif y[ts[i]]==-1 and pred[j][i]==1:
             FP.append(ts[i])      
    
    TPR=(len(TP)/(len(TP)+len(FN)))
    TNR=(len(TN)/(len(TN)+len(FP)))
    acc=(len(TN)+len(TP))/(len(TN)+len(TP)+len(FN)+len(FP))
    bacc=(TPR+TNR)/2
    auc=roc_auc_score(ytest,pred[j])
    gm=geometric_mean_score(ytest,pred[j])
    resg.append([TPR,TNR,acc,bacc,auc,gm])
  resg=np.array(resg)
  atpr=sum(resg[:,0])/len(pred);atnr=sum(resg[:,1])/len(pred)
  aac=sum(resg[:,2])/len(pred);abac=sum(resg[:,3])/len(pred) 
  aauc=sum(resg[:,4])/len(pred);agm=sum(resg[:,5])/len(pred)
  print('avg_TPR:',atpr,'avg_TNR:',atnr)
  print('avg_acc:',aac,'avg_Bacc:',abac)
  print('avg_auc:',aauc,'avg_GM:',agm)
  return [atpr,atnr,aac,abac,aauc,agm]



def get_fairness(sa_index,p_Group,in_ts,pred,X,y):
  results={}
  for s in range(len(sa_index)):
    resg,resr=[],[]
    for j in range(len(pred)):        
      ts=in_ts[j]
      TPp,TNp,FPp,FNp=[],[],[],[]
      TPu,TNu,FPu,FNu=[],[],[],[]
      for i in range(len(ts)):
        r=X[ts[i]]  
        if isinstance(p_Group[s],list)==False:
          if y[ts[i]]==1 and pred[j][i]==1:
            if r[sa_index[s]]<=p_Group[s]:
               TPp.append(ts[i])
            else:
               TPu.append(ts[i])
          elif y[ts[i]]==-1 and pred[j][i]==-1:
            if r[sa_index[s]]<=p_Group[s]:
               TNp.append(ts[i])
            else:
               TNu.append(ts[i])
          elif y[ts[i]]==1 and pred[j][i]==-1:
            if r[sa_index[s]]<=p_Group[s]:
               FNp.append(ts[i])
            else:
               FNu.append(ts[i])
          elif y[ts[i]]==-1 and pred[j][i]==1:
            if r[sa_index[s]]<=p_Group[s]:
               FPp.append(ts[i])
            else:
               FPu.append(ts[i])  
        elif isinstance(p_Group[s],list)==True:
          if y[ts[i]]==1 and pred[j][i]==1:
            if r[sa_index[s]]<=p_Group[s][0] or r[sa_index[s]]>=p_Group[s][1]:
               TPp.append(ts[i])
            else:
               TPu.append(ts[i])
          elif y[ts[i]]==-1 and pred[j][i]==-1:
            if r[sa_index[s]]<=p_Group[s][0] or r[sa_index[s]]>=p_Group[s][1]:
               TNp.append(ts[i])
            else:
               TNu.append(ts[i])
          elif y[ts[i]]==1 and pred[j][i]==-1:
            if r[sa_index[s]]<=p_Group[s][0] or r[sa_index[s]]>=p_Group[s][1]:
               FNp.append(ts[i])
            else:
               FNu.append(ts[i])
          elif y[ts[i]]==-1 and pred[j][i]==1:
            if r[sa_index[s]]<=p_Group[s][0] or r[sa_index[s]]>=p_Group[s][1]:
               FPp.append(ts[i])
            else:
               FPu.append(ts[i]) 
      TPR_p=(len(TPp)/(len(TPp)+len(FNp)))
      TNR_p=(len(TNp)/(len(TNp)+len(FPp)))
      TPR_u=(len(TPu)/(len(TPu)+len(FNu)))
      TNR_u=(len(TNu)/(len(TNu)+len(FPu)))
      resg.append([TPR_p,TNR_p])
      resr.append([TPR_u,TNR_u])
    resg=np.array(resg);resr=np.array(resr)
    atprg=sum(resg[:,0])/len(pred);atnrg=sum(resg[:,1])/len(pred)
    atprr=sum(resr[:,0])/len(pred);atnrr=sum(resr[:,1])/len(pred)
    print("\n\nFor Sensitive attribute index ",sa_index[s])  
    print('avg_TPR_unprot:',atprr,'avg_TPR_prot:',atprg)
    print('avg_TNR_unprot:',atnrr,'avg_TNR_prot:',atnrg)
    print("\n-------------------------------------------\n",)
    eqo=abs(atprr-atprg) + abs(atnrr - atnrg)
    results[sa_index[s]]=[atprg,atprr,atnrg,atnrr,eqo]
  return results  


def vis(path,results,prf=[],L=['Marital','Race','Sex','Age'],dt='Adult',clfs=['Multi-fair','Zafar et al.','AdaFair','FairLearn','MiniMax']):
    perfs=[]
    zz=results
    for i in range(len(zz)):
        c1,mxd=[],0
        for r in zz[i]:
            #c1.append(abs(r[0]-r[1]))
            mm=max(abs(r[0]-r[1]),abs(r[2]-r[3]))
            c1.append(abs(abs(r[0]-r[2])-abs(r[1]-r[3])))
            c1.append(r[4])
            if mm>mxd:
                mxd=mm
        c1.append(mxd)
       
        c1=c1+list(prf[i][:3])+list(prf[i][4:])
        perfs.append(c1)
    bar_width = 0.175
    labels=[]
    for i in L:
        #labels.append('dFNR')
        labels.append('CDM')
        labels.append('DM')
    labels.append('MMM')
    labels=labels+['TPR','TNR','Acc','AUC','G.Mean' ]
    print(labels)
    x = np.arange(len(labels))
    plt.figure(figsize=[30,10])
    plt.rcParams.update({'font.size': 25})
    plt.rc('font', weight='bold')
    plt.yticks(np.arange(0, 1.01, step=0.05))
    plt.ylim([0,1])
    plt.grid(True, axis='y')
    x = np.arange(len(labels))
    index = np.arange(0, len(labels)*1.3, step=1.3)
    plt.xticks(index + 1.5*bar_width ,labels,rotation=45)
    print(perfs)
    for i in range(0,len(clfs)):
        plt.bar(index + bar_width * i,perfs[i],bar_width,label=clfs[i])
    plt.legend(loc=2,ncol=1, shadow=False)
   
    #plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.draw()
    plt.savefig(path+dt+'_.png')
    plt.show()

