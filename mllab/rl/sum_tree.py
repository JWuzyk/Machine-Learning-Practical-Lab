import numpy as np


class SumTree:
    def __init__(self, capacity):
        # (capacity - 1) parent nodes, capactity leaf nodes
        assert capacity > 1
        self.capacity = capacity
        self._n_parents = capacity - 1
        self.tree = np.full(self._n_parents + capacity, np.nan, dtype='float64')
        self.tree[:self._n_parents] = 0.0
        self._len = 0
        self._min = (None, np.inf)  # (index, p)
        self._max = (None, -np.inf)  # (index, p)

    def add(self, p):
        """Add an element with weight p and return its index."""
        if self._len < self.capacity:
            index = self._len
            self._len += 1
        else:
            # Remove a smallest element
            index = self._min[0] - self._n_parents
        self.update(index, p)
        return index

    def update(self, index, p):
        """Update the element at given index."""
        assert index < self.capacity, "Index out of bound"
        tree_index = index + self.capacity - 1
        if np.isnan(self.tree[tree_index]):
            diff = p
        else:
            diff = p - self.tree[tree_index]
            if abs(diff) < 2e-7:
                return
        self.tree[tree_index] = p

        if self._min[0] == tree_index:
            # This is O(n), it takes a very few ms for one million
            argmin_index = self.tree[self._n_parents:self._n_parents + self._len].argmin() + self._n_parents
            self._min = (argmin_index, self.tree[argmin_index])
        if self._max[0] == tree_index:
            argmax_index = self.tree[self._n_parents:self._n_parents + self._len].argmax() + self._n_parents
            self._max = (argmax_index, self.tree[argmax_index])
        if p < self._min[1]:
            self._min = (tree_index, p)
        if p > self._max[1]:
            self._max = (tree_index, p)

        while True:
            parent = (tree_index - 1) // 2
            self.tree[parent] += diff
            if parent == 0:
                break
            tree_index = parent

    @property
    def total(self):
        return self.tree[0]

    @property
    def min(self):
        return None if self._min[0] is None else self._min[1]

    @property
    def max(self):
        return None if self._max[0] is None else self._max[1]

    def __len__(self):
        return self._len

    def find(self, p):
        """Get biggest index and s.t. the partial sum is below p."""
        tree_index = 0
        while True:
            left = 2 * tree_index + 1
            if left >= len(self) + self._n_parents:
                index = tree_index - self._n_parents
                return index, self.tree[tree_index]
            elif self.tree[left] >= p:
                tree_index = left
            elif left + 1 >= len(self) + self._n_parents:
                index = left - self._n_parents
                return index, self.tree[left]
            else:
                tree_index = left + 1  # right
                p -= self.tree[left]

    def __get__(self, index):
        assert index < self.capacity
        return self.tree[index + self._n_parents]


def test_add_and_min():
    st = SumTree(2)
    assert st.add(5) == 0
    assert st.min == 5
    assert st.add(2) == 1
    assert st.min == 2
    assert st.add(9) == 1
    assert st.min == 5


def test_minmax_after_self_update():
    st = SumTree(10)
    st.add(5)
    assert st.max == 5
    assert st.min == 5
    i = st.add(2)
    assert st.min == 2
    assert st.max == 5
    # update min
    st.update(i, 9)
    assert st.min == 5
    assert st.max == 9
    # update max
    st.update(i, 2)
    assert st.min == 2
    assert st.max == 5

def test_total():
    st = SumTree(3)
    assert st.total == 0

    st.add(1)
    assert st.total == 1

    st.add(2)
    assert st.total == 3

    st.add(4)
    assert st.add(5) == 0
    assert st.total == 11

    assert st.add(3) == 1
    assert st.total == 12


def test_add_0():
    st = SumTree(2)
    st.add(0)
    assert st.min == 0


def test_len():
    st = SumTree(2)
    assert len(st) == 0
    st.add(0)
    assert len(st) == 1
    st.add(0)
    assert len(st) == 2
    st.add(1)
    assert len(st) == 2

def test_find():
    #           0
    #         /   \
    #      10      5
    #     /  \    / \
    #   9     1  2   3
    #  / \
    # 4   5
    st = SumTree(5)
    st.add(1)
    st.add(2)
    st.add(3)
    st.add(4)
    st.add(5)
    assert st.find(1) == (3, 4.0)
    assert st.find(8.99) == (4, 5.0)
    assert st.find(9.5) == (0, 1.0)


def test_find_end():
    st = SumTree(4)
    st.add(1)
    st.add(2)
    st.add(3)
    assert st.find(10) == (2, 3.0)
