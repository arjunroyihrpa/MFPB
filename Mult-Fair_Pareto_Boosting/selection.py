"""
Adapted from the NSGA3 implementation [1] of pymoo [2].

[1]: https://github.com/msu-coinlab/pymoo/blob/c238f79587e1a1bb107efb893fe99aa043442e5a/pymoo/algorithms/nsga3.py
[2]: https://pymoo.org/


@author: Philip Naumann
"""

import warnings

import numpy as np
from numpy.linalg import LinAlgError

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.function_loader import load_function
from pymoo.util.misc import intersect, has_feasible


class PreferenceSurvival:
    def __init__(self, preference_vectors):
        # set random seed to avoid different outcomes
        np.random.seed(0)
        # ! Important
        # ? Is each solution feasible already? Is this is validated before?
        # super().__init__(filter_infeasible=True)
        self.ref_dirs = preference_vectors
        self.extreme_points = None
        self.intercepts = None
        self.nadir_point = None
        # self.opt = None
        self.ideal_point = np.full(preference_vectors.shape[1], np.inf)
        self.worst_point = np.full(preference_vectors.shape[1], -np.inf)

    def do(self, objective_values, n_survive=None, filter_duplicates=True):
        """
        Applies the non-dominated filtering based on the provided preference vectors.
        NDS is applied with respect to the preference vectors. I.e. each solution gets sorted
        along it's niche of the respective preference vector.

        Parameters
        ----------
        solutions : np.array
            Array matrix of all solutions in the COMPLETE final set.
            I.e., NDS has not been applied yet!
        objective_values : np.array
            Array matrix of the objective values for each of the solutions.
        n_survive : int
            Number of solutions that shall survive at most.
        filter_duplicates : bool
            Filter the final set of solutions if it contains duplicates.
            Else, at most as many as preference_vectors can be selected.

        Returns
        -------
        np.array
            NDS solutions with respect to the preference vectors.
            Is equal to at most the number of preference vectors we have.
        """
        # attributes to be set after the survival
        solutions = objective_values.copy()  # x, here it's also f(x)
        F = objective_values.copy()  # f(x)
        # compute the final solution set index in parallel
        running_index_opt = np.arange(len(F))
        running_index_survival = np.arange(len(F))

        if n_survive is None:
            # Select at most as many as we have solution candidates
            # ! This is not used anyway, as we don't need it for our purpose
            # ! so it can be disregarded and left as default value, i.e. None
            n_survive = len(F)

        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # calculate the fronts of the population
        # TODO maybe we should remove n_survive as we want to sort all anyway?
        fronts, rank = NonDominatedSorting().do(
            F, return_rank=True, n_stop_if_ranked=n_survive
        )
        non_dominated, last_front = fronts[0], fronts[-1]

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(
            F[non_dominated, :], self.ideal_point, extreme_points=self.extreme_points
        )

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[non_dominated, :], axis=0)

        self.nadir_point = get_nadir_point(
            self.extreme_points,
            self.ideal_point,
            self.worst_point,
            worst_of_population,
            worst_of_front,
        )

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        solutions, rank, F, running_index_opt, running_index_survival = (
            solutions[I],
            rank[I],
            F[I],
            running_index_opt[I],
            running_index_survival[I],
        )

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        niche_of_individuals, dist_to_niche, dist_matrix = associate_to_niches(
            F, self.ref_dirs, self.ideal_point, self.nadir_point
        )

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(
            dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0)
        )
        # * Select best solution per preference vector here
        intersection = intersect(fronts[0], closest)
        # ! This can be empty if the closest solutions are not in the non-dominated front
        # TODO potentially adjust this
        optimal_solutions_objective_values = solutions[intersection]
        running_index_opt = running_index_opt[intersection]

        # ! We don't need that as we are only interested in the best solution
        # ! per preference vector
        # * This could be used if we were interested to pick more solutions than the optimal ones
        # if we need to select individuals to survive
        if len(solutions) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=np.int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=np.int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(
                    len(self.ref_dirs), niche_of_individuals[until_last_front]
                )
                n_remaining = n_survive - len(until_last_front)

            S = niching(
                solutions[last_front],
                n_remaining,
                niche_count,
                niche_of_individuals[last_front],
                dist_to_niche[last_front],
            )

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            solutions = solutions[survivors]
            running_index_survival = running_index_survival[survivors]

        # ! Not stable currently
        # if filter_duplicates:
        #     optimal_solutions_objective_values, unique_idx = np.unique(
        #         optimal_solutions_objective_values, return_index=True, axis=0
        #     )
        #     running_index_opt = running_index_opt[unique_idx]

        return (
            solutions,
            running_index_survival,
            optimal_solutions_objective_values,
            running_index_opt,
        )


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / denom
    dist_matrix = load_function("calc_perpendicular_distance")(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche, dist_matrix


def niching(solutions, n_remaining, niche_count, niche_of_individuals, dist_to_niche):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(len(solutions), True)

    while len(survivors) < n_remaining:

        # number of individuals to select in this iteration
        n_select = n_remaining - len(survivors)

        # all niches where new individuals can be assigned to and the corresponding niche count
        next_niches_list = np.unique(niche_of_individuals[mask])
        next_niche_count = niche_count[next_niches_list]

        # the minimum niche count
        min_niche_count = next_niche_count.min()

        # all niches with the minimum niche count (truncate if randomly if more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[np.random.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(
                np.logical_and(niche_of_individuals == next_niche, mask)
            )[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            np.random.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            # add the selected individual to the survivors
            mask[next_ind] = False
            survivors.append(int(next_ind))

            # increase the corresponding niche count
            niche_count[next_niche] += 1

    return survivors


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=np.int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count


def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # add the old extreme points to never loose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points


def get_nadir_point(
    extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population
):
    try:

        # find the intercepts using gaussian elimination
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)

        warnings.simplefilter("ignore")
        intercepts = 1 / plane

        nadir_point = ideal_point + intercepts

        # check if the hyperplane makes sense
        if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6):
            raise LinAlgError()

        # if the nadir point should be larger than any value discovered so far set it to that value
        # NOTE: different to the proposed version in the paper
        b = nadir_point > worst_point
        nadir_point[b] = worst_point[b]

    except LinAlgError:

        # fall back to worst of front otherwise
        nadir_point = worst_of_front

    # if the range is too small set it to worst of population
    b = nadir_point - ideal_point <= 1e-6
    nadir_point[b] = worst_of_population[b]

    return nadir_point
