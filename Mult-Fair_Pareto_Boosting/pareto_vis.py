from Maximus_optimized_non_dominated import Multi_Fair as maximus
from sklearn.model_selection import StratifiedShuffleSplit as ss
from DataPreprocessing.my_utils import get_score, get_fairness, vis
import numpy as np

from DataPreprocessing.load_credit import load_credit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy as sp
import scipy.interpolate

from matplotlib.tri import Triangulation


def plot_pareto_front(
    associations, all_preference_vectors=None, full_pareto_front=None
):
    plt.clf()

    final_selected_solutions = np.row_stack([r[1] for r in associations])

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

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")

    # if we use final_selected_solutions for the max instead of
    # pareto_front, then we focus on the selected solutions, but
    # some of the complete pareto_front are not shown in the plot!
    # I think it makes sense this way though
    scale_factor_x = 1.0 / max(final_selected_solutions[:, 0])
    scale_factor_y = 1.0 / max(final_selected_solutions[:, 1])
    scale_factor_z = 1.0 / max(final_selected_solutions[:, 2])

    # scale_factor_x, scale_factor_y, scale_factor_z = (1.0, 1.0, 1.0)

    orig_used_ref_dirs = used_ref_dirs.copy().T

    used_ref_dirs[0, :] /= scale_factor_x
    used_ref_dirs[1, :] /= scale_factor_y
    used_ref_dirs[2, :] /= scale_factor_z

    unused_ref_dirs[0, :] /= scale_factor_x
    unused_ref_dirs[1, :] /= scale_factor_y
    unused_ref_dirs[2, :] /= scale_factor_z

    ax.set_xlim(0, 1 / scale_factor_x)
    ax.set_ylim(0, 1 / scale_factor_y)
    ax.set_zlim(0, 1 / scale_factor_z)
    ax.view_init(elev=20.0, azim=32)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ratio = 0.02

    print(used_ref_dirs)
    print(unused_ref_dirs)

    x, y, z = np.zeros((3, used_ref_dirs.shape[1]))
    u, v, w = used_ref_dirs
    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        arrow_length_ratio=ratio,
        color="blue",
        label="Used ref dirs",
        zorder=0,
    )

    x, y, z = np.zeros((3, unused_ref_dirs.shape[1]))
    u, v, w = unused_ref_dirs
    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        arrow_length_ratio=ratio,
        color="red",
        label="Unused ref dirs",
        zorder=0,
    )

    for i, row in enumerate(orig_used_ref_dirs):
        x, y, z = row
        label = "(%d, %d, %d)" % (x, y, z)
        xx, yy, zz = used_ref_dirs.T[i, :] / 2
        ax.text(xx, yy, zz, label)

    ax.scatter(
        xs=full_pareto_front[:, 0],
        ys=full_pareto_front[:, 1],
        zs=full_pareto_front[:, 2],
        label="Pareto-front",
        color="orange",
        s=35,
        zorder=1,
    )

    # x, y, z = full_pareto_front.T
    # triang = Triangulation(x, y)
    # ax.plot_trisurf(triang, z, color="green", shade=True, alpha=0.5)

    """
    x_grid = np.linspace(0, max(final_selected_solutions[:,0]), 1*len(x))
    y_grid = np.linspace(0, max(final_selected_solutions[:,1]), 1*len(y))
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')
    spline = sp.interpolate.Rbf(x,y,z,function='thin_plate',smooth=0.1)#, episilon=5)
    Z = spline(X,Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='winter', edgecolor='none')
    """
    """
    x_grid = np.linspace(0, max(final_selected_solutions[:,0]), 1*len(x))
    y_grid = np.linspace(0, max(final_selected_solutions[:,1]), 1*len(y))
    B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
    Z = np.zeros((x.size, z.size))
    spline = sp.interpolate.Rbf(x,y,z,function='thin_plate',smooth=0.1)#, episilon=5)
    Z = spline(B1,B2)
    ax.plot_wireframe(B1, B2, Z, zorder=1)
    ax.plot_surface(B1, B2, Z,alpha=0.2,zorder=1)
    """

    print(final_selected_solutions)
    final_selected_solutions = final_selected_solutions.T
    # ax.scatter(xs=final_selected_solutions[:,0],ys=final_selected_solutions[:,1],zs=final_selected_solutions[:,2],label="Pareto-front",
    #           color="k", s=35)
    x, y, z = np.zeros((3, final_selected_solutions.shape[1]))
    u, v, w = final_selected_solutions
    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        arrow_length_ratio=ratio,
        color="black",
        label="Selected solutions",
        zorder=2,
    )

    for i, row in enumerate(final_selected_solutions.T):
        associated_vector = used_ref_dirs.T[i, :]
        x, y, z = row / 2
        xx, yy, zz = associated_vector / 2
        ax.plot(
            [x, xx],
            [y, yy],
            [z, zz],
            color="green",
            linewidth=2,
            label="Association" if i == 0 else None,
            zorder=2,
        )

    plt.legend()
    # plt.savefig("pareto_front_with_ref_dirs.pdf")
    plt.show()


if __name__ == "__main__":
    X, y, sa_index, p_Group, x_control, F = load_credit()
    protected = [F[v] for v in sa_index]
    dt = "Credit"

    sss = ss(n_splits=2, test_size=0.4)  # for reporting experiments use n_splits=10
    preference = [
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
        [1.0, 0.05, 0.05],
    ]

    X = X[:5000, :]
    y = y[:5000]

    soln_per_dir = {}
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = maximus(
            n_estimators=499, saIndex=sa_index, saValue=p_Group, preference=preference
        )
        clf.fit(X_train, y_train)
        print(clf.preference_direction_to_solution_mapping)
        # caling clf.PF will give the pareto front
        for (a, b, c) in clf.preference_direction_to_solution_mapping:
            clf.estimators_ = clf.estimators_[:c]
            clf.estimator_alphas_ = clf.estimator_alphas_[:c]
            pred = clf.predict(X_test)
            # print(confusion_matrix(y_test,pred))
            if str(a) in soln_per_dir:
                soln_per_dir[str(a)].append([test_index, pred])
            else:
                soln_per_dir[str(a)] = []
                soln_per_dir[str(a)].append([test_index, pred])
    full_pareto_front = np.array(list(clf.PF.values()))
    plot_pareto_front(
        clf.preference_direction_to_solution_mapping,
        np.array(clf.preference),
        full_pareto_front,
    )
