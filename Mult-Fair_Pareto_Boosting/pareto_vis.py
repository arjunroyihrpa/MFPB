from Maximus_optimized_non_dominated import Multi_Fair as maximus
from sklearn.model_selection import StratifiedShuffleSplit as ss
from DataPreprocessing.my_utils import get_score,get_fairness,vis
import numpy as np

from DataPreprocessing.load_credit import load_credit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pareto_front(associations, all_preference_vectors=None):
    plt.clf()
    
    pareto_front = np.row_stack([r[1] for r in associations])
    
    used_ref_dirs = np.row_stack([r[0] for r in associations])
    unused_ref_dirs = []
    for row in all_preference_vectors:
        isin = False
        for used_row in used_ref_dirs:
            if np.array_equal(row, used_row):
                isin = True
        if not isin:    
            unused_ref_dirs.append(row)
    unused_ref_dirs = np.row_stack(unused_ref_dirs).T
    used_ref_dirs = used_ref_dirs.T
    
    fig = plt.figure(figsize=(8,10))
    ax = fig.add_subplot(111, projection='3d')
    scale_factor_x = 1.0 / max(pareto_front[:,0])
    scale_factor_y = 1.0 / max(pareto_front[:,1])
    scale_factor_z = 1.0 / max(pareto_front[:,2])
    
    orig_used_ref_dirs = used_ref_dirs.copy().T
    
    used_ref_dirs[0,:] /= scale_factor_x
    used_ref_dirs[1,:] /= scale_factor_y
    used_ref_dirs[2,:] /= scale_factor_z

    unused_ref_dirs[0,:] /= scale_factor_x
    unused_ref_dirs[1,:] /= scale_factor_y
    unused_ref_dirs[2,:] /= scale_factor_z
    
    ax.set_xlim(0,1/scale_factor_x)
    ax.set_ylim(0,1/scale_factor_y)
    ax.set_zlim(0,1/scale_factor_z)
    ax.view_init(elev=20., azim=32)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    ratio = 0.02
    
    print(used_ref_dirs)
    print(unused_ref_dirs)

    x, y, z = np.zeros((3,used_ref_dirs.shape[1]))
    u, v, w = used_ref_dirs
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=ratio,
              color="blue", label="Used ref dirs", zorder=0)
    
    x, y, z = np.zeros((3,unused_ref_dirs.shape[1]))
    u, v, w = unused_ref_dirs
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=ratio,
              color="red", label="Unused ref dirs", zorder=0)
    
    for i,row in enumerate(orig_used_ref_dirs):
        x,y,z = row
        label = '(%d, %d, %d)' % (x, y, z)
        xx, yy, zz = used_ref_dirs.T[i,:] / 2
        ax.text(xx, yy, zz, label)
    
    print(pareto_front)
    pareto_front = pareto_front.T
    #ax.scatter(xs=pareto_front[:,0],ys=pareto_front[:,1],zs=pareto_front[:,2],label="Pareto-front",
    #           color="k", s=35)
    x, y, z = np.zeros((3,pareto_front.shape[1]))
    u, v, w = pareto_front
    ax.quiver(x,y,z,u,v,w,arrow_length_ratio=ratio,
              color="black", label="Pareto-front", zorder=1)
    
    for i, row in enumerate(pareto_front.T):
        associated_vector = used_ref_dirs.T[i,:]
        x,y,z = row / 2
        xx,yy,zz = associated_vector / 2
        ax.plot([x,xx],[y,yy],[z,zz],color="green",linewidth=2, label="Association" if i==0 else None)
        
        
    
    plt.legend()
    # plt.savefig("pareto_front_with_ref_dirs.pdf")
    plt.show()
    


if __name__ == "__main__":
    X, y, sa_index, p_Group, x_control,F = load_credit()
    protected=[F[v] for v in sa_index]
    dt='Credit'

    sss = ss(n_splits=2,test_size=0.4) #for reporting experiments use n_splits=10
    preference=[[0,0,1],[0,1,1],[1,0,0],[1,1,0],[1,0,1],[0,1,0],[1.0,0.05,0.05]]

    X = X[:5000,:]
    y = y[:5000]

    soln_per_dir={}
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf=maximus(n_estimators=499,saIndex=sa_index,
            saValue=p_Group,
            preference=preference)
        clf.fit(X_train,y_train)
        print(clf.preference_direction_to_solution_mapping)
        # caling clf.PF will give the pareto front
        for (a,b,c) in clf.preference_direction_to_solution_mapping:
            clf.estimators_=clf.estimators_[:c]
            clf.estimator_alphas_=clf.estimator_alphas_[:c]
            pred=clf.predict(X_test)
            #print(confusion_matrix(y_test,pred))
            if str(a) in soln_per_dir:
                soln_per_dir[str(a)].append([test_index,pred])
            else:
                soln_per_dir[str(a)]=[]
                soln_per_dir[str(a)].append([test_index,pred])
    plot_pareto_front(clf.preference_direction_to_solution_mapping, np.array(clf.preference))