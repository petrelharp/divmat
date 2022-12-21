# Efficient computation of the pairwise divergence matrix.
import sys
import time
import itertools
import matplotlib.pyplot as plt

import msprime
import tskit
import numpy as np


def assert_dicts_close(d1, d2):
    if not set(d1.keys()) == set(d2.keys()):
        print("d1:", set(d1.keys()) - set(d2.keys()))
        print("d2:", set(d2.keys()) - set(d1.keys()))
    assert set(d1.keys()) == set(d2.keys())
    for k in d1:
        if not np.isclose(d1[k], d2[k]):
            print(k, d1[k], d2[k])
    for k in d1:
        assert np.isclose(d1[k], d2[k])


class DivergenceMatrix:
    def __init__(
        self,
        num_nodes,
        samples,
        nodes_time,
        edges_left,
        edges_right,
        edges_parent,
        edges_child,
        edge_insertion_order,
        edge_removal_order,
        sequence_length,
        num_trees = 0,
    ):
        # Quintuply linked tree
        self.parent = np.full(num_nodes + 1, -1, dtype=np.int32)
        self.left_sib = np.full(num_nodes + 1, -1, dtype=np.int32)
        self.right_sib = np.full(num_nodes + 1, -1, dtype=np.int32)
        self.left_child = np.full(num_nodes + 1, -1, dtype=np.int32)
        self.right_child = np.full(num_nodes + 1, -1, dtype=np.int32)
        # Sample lists refer to sample *index*
        self.sample_index_map = np.full(num_nodes + 1, -1, dtype=np.int32)
        self.num_samples = np.full(num_nodes + 1, 0, dtype=np.int32)
        # Edges and indexes
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edges_parent = edges_parent
        self.edges_child = edges_child
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.sequence_length = sequence_length
        self.nodes_time = nodes_time
        self.samples = samples
        self.position = 0
        self.virtual_root = num_nodes

        n = samples.shape[0]
        for j in range(n):
            u = samples[j]
            self.sample_index_map[u] = j
            self.num_samples[u] = 1
            self.insert_root(u)

        # this looks ugly for numba
        self.stack_u = [[np.int32(x) for x in range(0)] for _ in range(num_nodes + 1)]
        self.stack_z = [[np.float64(x) for x in range(0)] for _ in range(num_nodes + 1)]
        self.x = np.zeros(num_nodes + 1)

        self.num_trees = num_trees
        self.stack_history = np.zeros((num_trees + 2, num_nodes + 1))
        self.tn = 0
        self.num_additions = np.zeros(num_trees + 2)
        self.num_deletions = np.zeros(num_trees + 2)

    def print_state(self, msg=""):
        num_nodes = len(self.parent)
        print(f"..........{msg}................")
        print(f"position = {self.position}")
        for j in range(num_nodes):
            st = "NaN" if j >= self.virtual_root else f"{self.nodes_time[j]}"
            pt = "NaN" if self.parent[j] == tskit.NULL else f"{self.nodes_time[self.parent[j]]}"
            print(f"node {j} -> {self.parent[j]}: "
                  f"ns = {self.num_samples[j]}, "
                  f"z = ({pt} - {st})"
                  f" * ({self.position} - {self.x[j]})"
                  f" = {self.get_z(j)}")
            for u, z in zip(self.stack_u[j], self.stack_z[j]):
                print(f"   {(j, u)}: {z}")
        print(f"Virtual root: {self.virtual_root}")
        roots = []
        u = self.left_child[self.virtual_root]
        while u != tskit.NULL:
            roots.append(u)
            u = self.right_sib[u]
        print("Roots:", roots)
        print("Current state:")
        state = self.current_state()
        for k in state:
            print(f"   {k}: {state[k]}")

    def remove_branch(self, p, c):
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == -1:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == -1:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = -1
        self.left_sib[c] = -1
        self.right_sib[c] = -1

    def insert_branch(self, p, c):
        self.parent[c] = p
        u = self.right_child[p]
        if u == -1:
            self.left_child[p] = c
            self.left_sib[c] = -1
            self.right_sib[c] = -1
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = -1
        self.right_child[p] = c

    def remove_root(self, root):
        self.remove_branch(self.virtual_root, root)

    def insert_root(self, root):
        self.insert_branch(self.virtual_root, root)
        self.parent[root] = -1

    def remove_edge(self, p, c):
        assert p != -1
        self.remove_branch(p, c)
        # check for root changes
        u = p
        while u != tskit.NULL:
            path_end = u
            path_end_was_root = (self.num_samples[u] > 0)
            self.num_samples[u] -= self.num_samples[c]
            u = self.parent[u]
        if path_end_was_root and (self.num_samples[path_end] == 0):
            self.remove_root(path_end)
        if self.num_samples[c] > 0:
            self.insert_root(c)

    def insert_edge(self, p, c):
        assert p != -1
        assert self.parent[c] == -1, "contradictory edges"
        # check for root changes
        u = p
        while u != tskit.NULL:
            path_end = u
            path_end_was_root = (self.num_samples[u] > 0)
            self.num_samples[u] += self.num_samples[c]
            u = self.parent[u]
        if self.num_samples[c] > 0:
            self.remove_root(c)
        if (self.num_samples[path_end] > 0) and not path_end_was_root:
            self.insert_root(path_end)
        self.insert_branch(p, c)

    ######### begin stack stuff

    def _add_to_stack(self, u, v, z):
        ustack_u = self.stack_u[u]
        ustack_z = self.stack_z[u]
        append = True
        for j in range(len(ustack_u)):
            if ustack_u[j] == v:
                ustack_z[j] += z
                append = False
        if append:
            ustack_u.append(v)
            ustack_z.append(z)

    def add_to_stack(self, u, v, z):
        # note: having stack entries that are zero is important sometimes
        if self.num_trees > 0:
            self.num_additions[self.tn] += 1
        self._add_to_stack(u, v, z)
        self._add_to_stack(v, u, z)

    def empty_stack(self, n):
        for w in self.stack_u[n]:
            wstack_u = self.stack_u[w]
            for j in range(len(wstack_u) - 1, -1, -1):
                u = wstack_u[j]
                if u == n:
                    if self.num_trees > 0:
                        self.num_deletions[self.tn] += 1
                    del wstack_u[j]
                    del self.stack_z[w][j]
                    break
        self.stack_u[n].clear()
        self.stack_z[n].clear()

    def verify_zero_spine(self, n):
        """
        Verify that there are no contributions along the path
        from n up to the root. (should be true after clear_spine)
        """
        for z in self.stack_z[n]:
            assert z == 0
        while n != tskit.NULL:
            assert self.parent[n] == tskit.NULL or self.x[n] == self.position
            n = self.parent[n]

    def get_z(self, u):
        p = self.parent[u]
        return 0 if p == tskit.NULL else (
                (self.nodes_time[self.parent[u]] - self.nodes_time[u])
                * (self.position - self.x[u])
        )

    def mrca(self, a, b):
        # just used for `current_state`
        aa = [a]
        while a != tskit.NULL:
            a = self.parent[a]
            aa.append(a)
        while b not in aa:
            b = self.parent[b]
        return b

    def current_state(self, verbose=False):
        """
        Compute the current output, for debugging.
        NOTE that the path back to the roots of disconnected trees
        *still counts* for divergence *between* those trees!
        (In other words, disconnected trees act as if they are
        connected to a virtual root by a branch of length zero.)
        """
        if verbose: print("---------------")
        out = {(a, b) : 0 for a in self.samples for b in self.samples if a < b}
        for a in self.samples:
            for b in self.samples:
                if a < b:
                    k = tuple(sorted([a, b]))
                    m = self.mrca(a, b)
                    # edges on the path up from a
                    pa = a
                    while pa != m:
                        if verbose: print("edge:", k, pa, self.get_z(pa))
                        out[k] += self.get_z(pa)
                        pa = self.parent[pa]
                    # edges on the path up from b
                    pb = b
                    while pb != m:
                        if verbose: print("edge:", k, pb, self.get_z(pb))
                        out[k] += self.get_z(pb)
                        pb = self.parent[pb]
                    # pairwise stack references along the way
                    pa = a
                    while pa != m:
                        pb = b
                        while pb != m:
                            for w, z in zip(self.stack_u[pa], self.stack_z[pa]):
                                if w == pb:
                                    if verbose: print("stack:", k, (pa, pb), z)
                                    out[k] += z
                            pb = self.parent[pb]
                        pa = self.parent[pa]
        if verbose: print("---------------")
        return out

    def clear_children(self, n):
        """
        Moves all contributions of the edges between n and the children of n
        into stack entries, also adding these edges to stack entries of n.
        This is only a valid move in the context of clear_spine,
        when the only stack entries are to sibs of the path back to the root.
        """
        # this operation should not change the current output
        # before_state = self.current_state()
        for w, z in zip(self.stack_u[n], self.stack_z[n]):
            c = self.left_child[n]
            while c != tskit.NULL:
                zc = self.get_z(c)
                # print(f"adding {z}+{zc}={z+zc} to {(w, c)}")
                self.add_to_stack(w, c, z + zc)
                c = self.right_sib[c]
        self.empty_stack(n)
        # self.print_state(f'after stack {n}')
        c = self.left_child[n]
        while c != tskit.NULL:
            zc = self.get_z(c)
            v = self.left_child[n]
            while v != tskit.NULL:
                if c != v:
                    # print(f"adding {zu} to {(c, v)}")
                    self.add_to_stack(c, v, zc)
                v = self.right_sib[v]
            self.x[c] = self.position
            c = self.right_sib[c]
        # after_state = self.current_state()
        # assert_dicts_close(before_state, after_state)

    def clear_node_stack(self, n):
        """
        Push down references in the stack from n to other nodes
        to the children of n.
        """
        for w, z in zip(self.stack_u[n], self.stack_z[n]):
            c = self.left_child[n]
            while c != tskit.NULL:
                # print(f"adding {z} to {(w, c)}")
                self.add_to_stack(w, c, z)
                c = self.right_sib[c]
        self.empty_stack(n)

    def clear_spine(self, n):
        """
        Clears all nodes on the path from n up to the root,
        by pushing the contributions of all their branches to the stack
        and pushing all stack references to their children.
        """
        # this operation should not change the current output
        # before_state = self.current_state()
        spine = []
        p = n
        while p != tskit.NULL:
            spine.append(p)
            p = self.parent[p]
        spine.append(self.virtual_root)
        # first clear existing stack entries
        for j in range(len(spine) - 1, -1, -1):
            self.clear_node_stack(spine[j])
        # then go through and make new entries for the edges;
        # this requires a different update step for the stack
        # entries made to propagate edges down the spine,
        # which is why previous ones had to be cleared first
        for j in range(len(spine) - 1, -1, -1):
            self.clear_children(spine[j])
        # self.verify_zero_spine(n)
        # after_state = self.current_state()
        # assert_dicts_close(before_state, after_state)

    def clear_subtree_stack(self, n):
        # this operation should not change the current output
        # before_state = self.current_state()
        self.clear_node_stack(n)
        # after_state = self.current_state()
        # assert_dicts_close(before_state, after_state)
        c = self.left_child[n]
        while c != -1:
            # TODO: this only works with samples all at time 0
            if self.nodes_time[c] > 0:
                self.clear_subtree_stack(c)
            c = self.right_sib[c]

    def clear_subtree_edges(self, n):
        # this operation should not change the current output
        # before_state = self.current_state()
        self.clear_children(n)
        # after_state = self.current_state()
        # assert_dicts_close(before_state, after_state)
        c = self.left_child[n]
        while c != -1:
            # TODO: this only works with samples all at time 0
            if self.nodes_time[c] > 0:
                self.clear_subtree_edges(c)
            c = self.right_sib[c]

    def run(self):
        sequence_length = self.sequence_length
        M = self.edges_left.shape[0]
        in_order = self.edge_insertion_order
        out_order = self.edge_removal_order
        edges_left = self.edges_left
        edges_right = self.edges_right
        edges_parent = self.edges_parent
        edges_child = self.edges_child

        j = 0
        k = 0
        # TODO: self.position is redundant with left
        self.position = left = 0

        while k < M and left < self.sequence_length:
            while k < M and edges_right[out_order[k]] == left:
                p = edges_parent[out_order[k]]
                c = edges_child[out_order[k]]
                self.clear_spine(p)
                assert self.x[c] == self.position
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.remove_edge(p, c)
                k += 1
                # self.print_state(f"remove {p, c}") ##
            while j < M and edges_left[in_order[j]] == left:
                p = edges_parent[in_order[j]]
                c = edges_child[in_order[j]]
                if self.position > 0:
                    self.clear_spine(p)
                assert self.parent[p] == tskit.NULL or self.x[p] == self.position
                self.insert_edge(p, c)
                self.x[c] = self.position
                j += 1
                # self.print_state(f"add {p, c}") ##
            right = sequence_length
            if j < M:
                right = min(right, edges_left[in_order[j]])
            if k < M:
                right = min(right, edges_right[out_order[k]])
            self.position = left = right
            left = right
            # stack recording
            if self.num_trees > 0:
                self.stack_history[self.tn, :] = [
                    len(u) for u in self.stack_u
                ]
            self.tn += 1
        # clear out final tree
        assert j == M and left == self.sequence_length
        self.clear_subtree_stack(self.virtual_root)
        self.clear_subtree_edges(self.virtual_root)
        out = np.zeros((len(self.samples), len(self.samples)))
        for i in self.samples:
            for j, z in zip(self.stack_u[i], self.stack_z[i]):
                out[i, j] = z
        return out


def divergence_matrix(ts):
    dm = DivergenceMatrix(
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
    )
    return dm.run()


def lib_divergence_matrix(ts, mode="branch"):
    out = ts.divergence(
            [[u] for u in ts.samples()],
            [(i, j) for i in range(ts.num_samples) for j in range(ts.num_samples)],
            mode=mode,
            span_normalise=False
    ).reshape((ts.num_samples, ts.num_samples))
    for i in range(ts.num_samples):
        out[i, i] = 0
    return out


def get_stack_history(ts):
    dm = DivergenceMatrix(
        ts.num_nodes,
        samples=ts.samples(),
        nodes_time=ts.nodes_time,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edges_parent=ts.edges_parent,
        edges_child=ts.edges_child,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
        sequence_length=ts.sequence_length,
        num_trees=ts.num_trees,
    )
    _ = dm.run()
    return dm


def verify():
    for seed in range(1, 10):
        ts = msprime.sim_ancestry(
                10,
                ploidy=1,
                population_size=10,
                sequence_length=100,
                recombination_rate=0.01,
                random_seed=seed
        )
        D1 = lib_divergence_matrix(ts, mode="branch")
        D2 = divergence_matrix(ts)
        assert np.allclose(D1, D2)


def plot_stack():
    seed = 123
    n = 20
    ts = msprime.sim_ancestry(
        n,
        ploidy=1,
        population_size=10**4,
        sequence_length=1e5,
        recombination_rate=1e-8,
        random_seed=seed,
    )
    dm = get_stack_history(ts)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(18,12))
    ax0.plot(dm.stack_history)
    ax0.set_title(f"num trees: {ts.num_trees}, num samples: {ts.num_samples}")
    ax0.set_xlabel("tree number")
    ax0.set_ylabel("stack depth")
    ax1.plot(np.sum(dm.stack_history, axis=1))
    ax1.set_xlabel("tree number")
    ax1.set_ylabel("total stack size")
    ax2.plot(dm.num_additions, label='stack additions')
    ax2.plot(dm.num_deletions, label='stack deletions')
    ax2.set_xlabel("tree number")
    ax2.set_ylabel("number of operations")
    ax2.legend()
    fig.savefig(f"stack_profile_{ts.num_trees}_{ts.num_samples}_{seed}.png", bbox_inches = "tight")


def plot_stack_growth():
    seed = 123
    nvals = np.linspace(4, 104, 21).astype("int")
    total_ops = np.zeros(len(nvals))
    treewise_ops = np.zeros(len(nvals))
    max_stack = np.zeros(len(nvals))
    num_trees = np.zeros(len(nvals))
    for j, n in enumerate(nvals):
        ts = msprime.sim_ancestry(
            n,
            ploidy=1,
            population_size=10**4,
            sequence_length=1e7,
            recombination_rate=1e-8,
            random_seed=seed,
        )
        num_trees[j] = ts.num_trees
        dm = get_stack_history(ts)
        total_ops[j] = np.sum(dm.num_additions) + np.sum(dm.num_deletions)
        max_stack[j] = np.max(np.sum(dm.stack_history, axis=1))
        for t in ts.trees():
            for n in t.nodes():
                treewise_ops[j] += t.num_samples(n)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12,12))
    ax0.scatter(nvals, total_ops, label=f"total operations")
    ax0.scatter(nvals, treewise_ops, label="treewise")
    ax0.set_xlabel("number of samples")
    ax0.set_ylabel("total number of operations")
    ax0.set_title("total number of operations")
    ax0.legend()
    ax1.scatter(num_trees, total_ops, label=f"total operations")
    ax1.scatter(num_trees, treewise_ops, label="treewise")
    ax1.set_xlabel("number of trees")
    ax1.set_ylabel("total number of operations")
    ax1.set_title("total number of operations")
    ax1.legend()
    ax2.scatter(nvals, max_stack, label="max stack size")
    ax2.set_xlabel("number of samples")
    ax2.set_ylabel("max stack size")
    ax2.set_title("max stack size")
    plt.tight_layout()
    fig.savefig(f"stack_history.png", bbox_inches = "tight")



if __name__ == "__main__":

    np.set_printoptions(linewidth=500, precision=4)
    verify()
    plot_stack()
    plot_stack_growth()
