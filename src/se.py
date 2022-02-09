"""
Implementations of the SE-A, SE-B and SE-C graph generators and some auxiliary utilities.
"""

from __future__ import annotations

import itertools
from random import Random
from typing import Iterator, Iterable, Callable, TypeVar, Union

import networkx as nx

T = TypeVar('T')


def se_a(n: int, rng: Random) -> nx.Graph:
    """
    Returns a random graph using the SE-A preferential attachment algorithm.

    This algorithm generates an undirected, unweighted graph, without self loops or multiple edges.

    :param int n: Number of nodes of the final graph.
    :param Random rng: Random number generator.
    :raises ValueError: If `n` is less than 2.
    :return: The resulting graph.
    :rtype: Graph
    """
    # Fail if n < 2
    if n < 2:
        raise ValueError(f"n must be greater or equal to 2, given n = {n}")

    # The sample set that stores the edges in a list
    edge_list: list[tuple[int, int]] = [(0, 1)]

    # The output graph which starts with two vertices and on edge between them
    g: nx.Graph = nx.Graph(edge_list)

    # Grow the graph until there are n vertices
    source: int = 2
    while source < n:
        # Select one random edge from the edge list
        random_edge: tuple[int, int] = rng.choice(edge_list)
        # Add two edges in the graph based on the new vertex and the randomly selected edge
        # Here we store the new edges in a list so we can add reuse the iterator of the zip operation
        new_edges: list[tuple[int, int]] = list(zip([source] * 2, random_edge))
        edge_list.extend(new_edges)
        g.add_edges_from(new_edges)
        # Advance source
        source += 1

    # Return the graph and we're done
    return g


def se_b(n: int, m: int, mu: int, rng: Random, initial_graph: nx.Graph = None) -> nx.Graph:
    r"""
    Returns a random graph using the SE-B preferential attachment algorithm.

    This algorithm generates an undirected, unweighted graph, without self loops or multiple edges.

    :param int n: Number of nodes of the final graph.
    :param int m: Number of edges to attach from a new node to existing nodes.
    :param int mu: An integer denoting the number of hyperedges that are shuffled before selecting.
    :param Random rng: Random number generator.
    :param Optional[Graph] initial_graph: Initial network for the algorithm. It must be an undirected graph without self
        loops of multiple edges. The initial graph must satisfy the divisibility :math:`2|E_0|/m` and no vertex can have
        degree higher than :math:`2|E_0|/m`. It should be connected, although this is not enforced. The initial graph
        will be copied before being used. This argument is optional and if `None` is given then the process starts from
        a complete graph of :math:`m` nodes and :math:`m(m-1)` edges.
    :raises ValueError: If :math:`n \ge |V_0| \ge m \ge 2` is not satisfied or if the conditions required for the
        `initial_graph` are not fulfilled, namely the divisibility :math:`2|E_0|/m` and the maximum degree of
        :math:`2|E_0|/m`.
    :return: The resulting graph.
    :rtype: Graph
    """
    if not m >= 2:
        raise ValueError(
            f"Condition m >= 2 is not met, got m = {m}"
        )

    if initial_graph is None:
        initial_graph = nx.complete_graph(m, nx.Graph)

    if not n >= len(initial_graph) >= m >= 2:
        raise ValueError(
            f"Condition n >= |V_0| >= m >= 2 is not met, got n = {n}, m = {m}, |V_0| = {len(initial_graph)}"
        )

    if not mu >= 1:
        raise ValueError(f"mu must be strictly positive, got mu = {mu}")

    # Create the initial hyperedge list
    hyperedge_list: list[set[object]] = RandomSystematicPartitioning(m, rng) \
        .add_items(initial_graph.nodes, lambda x: initial_graph.degree[x]) \
        .partition()

    # Initialize the graph
    g = initial_graph.copy()

    # Grow the graph until there are n vertices
    source: int = len(g)
    while source < n:
        # Check if the source vertex is already in the graph. This happens when the initial graph has inconsistent
        # vertex labels, for example contains integer labels greater than the size of the initial graph.
        if g.has_node(source):
            raise ValueError(f"The initial graph contains node {source} already")
        # Create two new hyperedges and add the source on both.
        # Here we also create an alternating iterator for convenience.
        hyperedge_x: set[object] = {source}
        hyperedge_y: set[object] = {source}
        new_hyperedges: Iterator[set[object]] = itertools.cycle([hyperedge_x, hyperedge_y])
        # Select a random hyperedge.
        random_hyperedge: set[object] = RandomSystematicPartitioning(m, rng).add_iterator(
            itertools.chain(*map(lambda x: hyperedge_list[x], random_choices(len(hyperedge_list), mu, rng)))
        ).sample()
        # Add its elements into the new hyperedges using a random split.
        # At the same time add the new edges in the graph.
        for v in shuffled(random_hyperedge, rng):
            next(new_hyperedges).add(v)
            g.add_edge(source, v)
        # Select m-2 hyperedges so we can swap the m-2 values of source that are remaining. One of these m-2 hyperedges
        # might be the random hyperedge previously selected.
        for h in random_selections(len(hyperedge_list), m - 2, rng):
            # Find a single element in the hyperedge h that can be inserted into the new hyperedge. For absolute
            # randomness, we iterate through the hyperedge in random order, although this step might not even be
            # necessary.
            current_hyperedge: set[object] = next(new_hyperedges)
            for v in shuffled(hyperedge_list[h], rng):
                if v not in current_hyperedge:
                    # When we find one vertex, push it to the current new hyperedge, remove it from the old hyperedge,
                    # and add source to the old hyperedge.
                    current_hyperedge.add(v)
                    hyperedge_list[h].remove(v)
                    hyperedge_list[h].add(source)
                    break

        # Add the new hyperedges on the hyperedge list
        hyperedge_list.append(next(new_hyperedges))
        hyperedge_list.append(next(new_hyperedges))
        # Advance source
        source += 1

    # Return the graph and we're done
    return g


def se_c(n: int, m: int, mu: int, rng: Random, initial_graph: nx.Graph = None) -> nx.Graph:
    r"""
    Returns a random graph using the SE-C preferential attachment algorithm.

    This algorithm generates an undirected, unweighted graph, without self loops or multiple edges.

    :param int n: Number of nodes of the final graph.
    :param int m: Number of edges to attach from a new node to existing nodes.
    :param int mu: An integer denoting the number of hyperedges that are shuffled before selecting.
    :param Random rng: Random number generator.
    :param Optional[Graph] initial_graph: Initial network for the algorithm. It must be an undirected graph without self
        loops of multiple edges. The initial graph must satisfy the divisibility :math:`2|E_0|/m` and no vertex can have
        degree higher than :math:`2|E_0|/m`. It should be connected, although this is not enforced. The initial graph
        will be copied before being used. This argument is optional and if `None` is given then the process starts from
        a complete graph of :math:`m` nodes and :math:`m(m-1)` edges.
    :raises ValueError: If :math:`n \ge |V_0| \ge m \ge 2` is not satisfied or if the conditions required for the
        `initial_graph` are not fulfilled, namely the divisibility :math:`2|E_0|/m` and the maximum degree of
        :math:`2|E_0|/m`.
    :return: The resulting graph.
    :rtype: Graph
    """
    if not m >= 2:
        raise ValueError(
            f"Condition m >= 2 is not met, got m = {m}"
        )

    if initial_graph is None:
        initial_graph = nx.complete_graph(m, nx.Graph)

    if not n >= len(initial_graph) >= m >= 2:
        raise ValueError(
            f"Condition n >= |V_0| >= m >= 2 is not met, got n = {n}, m = {m}, |V_0| = {len(initial_graph)}"
        )

    if not mu >= 1:
        raise ValueError(f"mu must be strictly positive, got mu = {mu}")

    # Create the initial hyperedge list
    hyperedge_list: list[set[object]] = RandomSystematicPartitioning(m, rng) \
        .add_items(initial_graph.nodes, lambda x: initial_graph.degree[x]) \
        .partition()

    # Initialize the graph
    g = initial_graph.copy()

    # Grow the graph until there are n vertices
    source: int = len(g)
    while source < n:
        # Check if the source vertex is already in the graph. This happens when the initial graph has inconsistent
        # vertex labels, for example contains integer labels greater than the size of the initial graph.
        if g.has_node(source):
            raise ValueError(f"The initial graph contains node {source} already")
        # Start the random systematic partitioning
        rsp: RandomSystematicPartitioning = RandomSystematicPartitioning(m, rng)
        # Select one random old hyperedge.
        random_hyperedge: set[object] = RandomSystematicPartitioning(m, rng).add_iterator(
            itertools.chain(*map(lambda x: hyperedge_list[x], random_choices(len(hyperedge_list), mu, rng)))
        ).sample()
        # Insert its elements into the RSP. At the same time create the edges too.
        for v in shuffled(random_hyperedge, rng):
            rsp.add_item(v, 1)
            g.add_edge(source, v)
        # Insert m copies of source into the RSP
        rsp.add_item(source, m)
        # Randomly select m-2 old hyperedges and add them into the RSP
        random_old_hyperedges: list[int] = list(random_selections(len(hyperedge_list), m - 2, rng))
        for i in random_old_hyperedges:
            rsp.add_iterator(hyperedge_list[i])
        # Run the partitioning in the RSP
        rsp_iter: Iterator[set[object]] = iter(rsp.partition())
        # Replace m-2 rows of the RSP into the random m-2 old hyperedges previously selected and insert the other 2
        for i in random_old_hyperedges:
            hyperedge_list[i] = next(rsp_iter)
        hyperedge_list.append(next(rsp_iter))
        hyperedge_list.append(next(rsp_iter))
        # Advance source
        source += 1

    # Return the graph and we're done
    return g


def se_d(n: int, m: int, rng: Random, initial_graph: nx.Graph = None) -> nx.Graph:
    if not m >= 2:
        raise ValueError(
            f"Condition m >= 2 is not met, got m = {m}"
        )

    if initial_graph is None:
        initial_graph = nx.complete_graph(m, nx.Graph)

    if not n >= len(initial_graph) >= m >= 2:
        raise ValueError(
            f"Condition n >= |V_0| >= m >= 2 is not met, got n = {n}, m = {m}, |V_0| = {len(initial_graph)}"
        )

    # Create the initial hyperedge list
    hyperedge_list: list[set[object]] = RandomSystematicPartitioning(m, rng) \
        .add_items(initial_graph.nodes, lambda x: initial_graph.degree[x]) \
        .partition()

    # Initialize the graph
    g = initial_graph.copy()

    # Grow the graph until there are n vertices
    source: int = len(g)
    while source < n:
        # Check if the source vertex is already in the graph. This happens when the initial graph has inconsistent
        # vertex labels, for example contains integer labels greater than the size of the initial graph.
        if g.has_node(source):
            raise ValueError(f"The initial graph contains node {source} already")
        # Start the random systematic partitioning
        rsp: RandomSystematicPartitioning = RandomSystematicPartitioning(m, rng)
        # Select one random old hyperedge.
        random_hyperedge: set[object] = RandomSystematicPartitioning(m, rng).add_iterator(
            itertools.chain(*hyperedge_list)
        ).sample()
        # Insert its elements into the RSP. At the same time create the edges too.
        for v in shuffled(random_hyperedge, rng):
            rsp.add_item(v, 1)
            g.add_edge(source, v)
        # Insert m copies of source into the RSP
        rsp.add_item(source, m)
        # Randomly select m-2 old hyperedges and add them into the RSP
        random_old_hyperedges: list[int] = list(random_selections(len(hyperedge_list), m - 2, rng))
        for i in random_old_hyperedges:
            rsp.add_iterator(hyperedge_list[i])
        # Run the partitioning in the RSP
        rsp_iter: Iterator[set[object]] = iter(rsp.partition())
        # Replace m-2 rows of the RSP into the random m-2 old hyperedges previously selected and insert the other 2
        for i in random_old_hyperedges:
            hyperedge_list[i] = next(rsp_iter)
        hyperedge_list.append(next(rsp_iter))
        hyperedge_list.append(next(rsp_iter))
        # Advance source
        source += 1

    # Return the graph and we're done
    return g


def shuffled(a: Iterable[T], rng: Random) -> Iterator[T]:
    """
    Returns a generator that iterates through the values of the input iterable in random order.

    This function will make a copy of the input iterable before using it. It is useful when the input is not a random
    access collection, for example a set, and you don't necessarily need to have the items in a concrete collection, for
    example you wish to perform a one-pass operation on them, for example you are using it as intermediate storage. In
    other cases, it probably won't be any good in terms of performance.

    :param Iterable[object] a: The input iterable.
    :param Random rng: The random number generator.
    :return: A generator for the values of the input collection in random order.
    :rtype: Iterator[object]
    """
    a: list[object] = list(a)
    for i in range(0, len(a) - 1):
        j: int = rng.randrange(i, len(a))
        yield a[j]
        a[j] = a[i]
    yield a[-1]


def random_selections(n: int, k: int, rng: Random) -> Iterator[int]:
    r"""
    Performs an unweighted selection without replacement of :math:`k` elements from a population of :math:`n` elements.

    The population and the sample are represented by their indices and, as a result, this method will return
    :math:`k` random and discrete indices in the range :math:`[0,n)`. The selection is performed in such a way that the
    higher order inclusion probabilities of all :math:`k`-tuples are equal. In practice, this means that if
    :math:`k = 2`, all pairs of numbers are equally likely to appear as the result of this method. The operation of this
    algorithm is based on a virtual shuffling of the array :math:`[0,n)` where the first :math:`k` elements are then
    being returned efficiently.

    If you need to reuse the result of this operation you need to store it in a collection. A :class:`set` would be a
    convenient container as the elements returned are unique and don't have any particular order at which they are
    returned:

    .. code-block:: python

       random_numbers = set(random_selections(10, 3, Random()))

    This method returns a generator that will be fully consumed in time proportional to :math:`k` in the worst case and
    is not prone to rejections due to the selection of duplicate elements. The generator uses memory proportional to
    :math:`k` in the worst case.

    :param int n: The size of the population.
    :param int k: The size of the sample.
    :param Random rng: The random number generator to use.
    :raises ValueError: If the condition :math:`n \ge k \ge 0` is not satisfied.
    :return: A generator that holds the values of :math:`k` random and discrete integers in the range :math:`[0,n)`.
    :rtype: Iterator[int]
    """
    # Conditions for the arguments
    if not (n >= k >= 0):
        raise ValueError(f"The condition n >= k >= 0 is not satisfied, got n = {n}, k = {k}")

    # Setup virtual swaps
    # Each time an element is being selected, we virtually swap it with the last element and remove the last element.
    # This process is performed so that the array [0,n) is not stored into memory which is very helpful when the value
    # of k is small in respect to n.
    swaps: dict[int, int] = dict()
    for i in range(k):
        next_index: int = rng.randrange(0, n - i)
        yield swaps.get(next_index, next_index)
        swaps[next_index] = swaps.get(n - i - 1, n - i - 1)


def random_choices(n: int, k: int, rng: Random) -> Iterator[int]:
    r"""
    Performs an unweighted selection with replacement of :math:`k` elements from a population of :math:`n` elements.

    The population and the sample are represented by their indices and, as a result, this method will return
    :math:`k` random (but not necessarily discrete) indices in the range :math:`[0,n)`. The selection is performed
    by repeated independent random selections from that range. The elements are returned in no particular order. This
    method is similar in operation with :func:`random.choices` but returns the choices as an iterator that is lazily
    populated instead of a concrete collection.

    If you need to reuse the result of this operation you need to store it in a collection. A :class:`list` must be used
    in this case as the elements returned might not be unique:

    .. code-block:: python

       random_numbers = list(random_choices(10, 3, Random()))

    This method returns a generator that will be fully consumed in time proportional to :math:`k` in the worst case and
    does not consume additional memory.

    :param int n: The size of the population.
    :param int k: The size of the sample.
    :param Random rng: The random number generator to use.
    :raises ValueError: If the conditions :math:`n \ge 0` and :math:`k \ge 0` are not satisfied.
    :return: A generator that holds the values of :math:`k` random and discrete integers in the range :math:`[0,n)`.
    :rtype: Iterator[int]
    """
    # Conditions for the arguments
    if not (n >= 0 and k >= 0):
        raise ValueError(f"The conditions n >= 0 and k >= 0 are not both satisfied, got n = {n}, k = {k}")

    # Iterator
    for _ in range(k):
        yield rng.randrange(0, n)


class RandomSystematicPartitioning:
    """
    Implementation of the random systematic partitioning scheme.

    Random systematic partitioning (RSP) randomly partitions the input collection into groups of :math:`k` elements such
    that each group contains only unique elements. The number of groups itself is implicitly defined by the ratio of the
    sum of frequencies over :math:`k`. The algorithm implemented in this class is heavily influenced by the systematic
    random sampling design. If the elements cannot be partitioned in such a way, then :class:`ValueError` is raised from
    the :meth:`~se.RandomSystematicPartitioning.partition` method.

    The interface allows you to add elements into the RSP using the methods
    :meth:`~se.RandomSystematicPartitioning.add_item` and :meth:`~se.RandomSystematicPartitioning.add_items` and then it
    is possible to use the method :meth:`~se.RandomSystematicPartitioning.partition` to do the actual partitioning. By
    itself the :meth:`~se.RandomSystematicPartitioning.partition` does not change the state of the instance.

    Below is a typical use case that will partition 2 copies of the element 'a' and the elements 'b' and 'c' into groups
    of 2.

    .. code-block:: python

       rsp = RandomSystematicPartitioning(2, random.Random())
       rsp.add_item('a', 2)
       rsp.add_item('b', 1)
       rsp.add_item('c', 1)
       print(rsp.partition())

    Because the element 'a' exists twice in the input collection, the only possible partitioning is the two groups
    (a, b) and (a, c). The actual order of the groups returned is not important as the elements are randomly shuffled
    anyway. In fact, the probability of receiving any of the two partitions is 50%. Keep in mind that the
    :meth:`~se.RandomSystematicPartitioning.partition` method is deterministic and will return the same result on
    consecutive calls. The actual randomization (or shuffling) is performed online as elements enter the instance.
    Statistically independent partitions (even of the same input data) must rely on different instances of this class
    and not different invocations of the :meth:`~se.RandomSystematicPartitioning.partition` method. As a result, calling
    :meth:`~se.RandomSystematicPartitioning.partition` is normally not required or needed.

    The :meth:`~se.RandomSystematicPartitioning.add_items` may be used to insert items into the RSP when the population
    is known and exists in a concrete collection and at the same time a mapping function exists that maps the elements
    with their frequencies. In a simple case where the elements along with their frequencies are stored in a dictionary
    `dict[object, int]` a typical use case would be the following:

    .. code-block:: python

       frequencies = {'a': 2, 'b': 1, 'c': 1}
       rsp = RandomSystematicPartitioning(2, random.Random())
       rsp.add_items(frequencies.keys(), lambda x: frequencies[x])
       print(rsp.partition())

    The interface of this class also allows the use of builder-like syntax because the `add_*` methods return the
    instance itself, for example:

    .. code-block:: python

       rsp = RandomSystematicPartitioning(2, random.Random())
       rsp.add_item('a', 2)
          .add_item('b', 1)
          .add_item('c', 1)
       print(rsp.partition())
    """

    def __init__(self, k: int, rng: Random) -> None:
        """
        The constructor initializes the instance with a group size `k` and a random number generator `rng`.

        The constructor runs in constant time.

        :param int k: The size of each group.
        :param Random rng: The random number generator.
        :raises ValueError: If :math:`k < 1`.
        """
        if k < 1:
            raise ValueError("k cannot be less than 1")

        self.__k: int = k
        self.__rng: Random = rng
        self.__frequencies: dict[object, int] = dict()
        self.__items: list[object] = list()
        self.__n: int = 0

    def add_item(self, item: object, frequency: int) -> RandomSystematicPartitioning:
        """
        Adds an item along with its frequency into the instance.

        If the item is already present in the instance as a result of a previous insertion, its frequency will be
        increased by `frequency`, otherwise it will be inserted and its frequency will be set to `frequency`. The
        elements are inserted into a :class:`dict`, and as a result the equality is performed using the `__hash__`
        and `__eq__` combination.

        Also see the :meth:`add_items` and :meth:`add_iterator` methods.

        :param object item: The item to insert.
        :param int frequency: The frequency of the item.
        :raises ValueError: If `frequency < 1`.
        :return: The instance itself (self).
        :rtype: RandomSystematicPartitioning
        """
        if frequency < 1:
            raise ValueError(f"frequency cannot be less than 1, got {frequency}")
        self.__n += frequency
        if item in self.__frequencies:
            self.__frequencies[item] += frequency
        else:
            self.__frequencies[item] = frequency
            self.__items.append(item)
            random_index: int = self.__rng.randrange(0, len(self.__items))
            self.__items[-1], self.__items[random_index] = self.__items[random_index], self.__items[-1]
        return self

    def add_items(self, items: Iterable[object],
                  mapping: Callable[[object], int]) -> RandomSystematicPartitioning:
        """
        Adds a collection of items along with their frequencies into the instance.

        This method is equivalent to:

        .. code-block:: python

           for v in items:
               self.add_item(v, mapping(v))
           return self

        The elements in `items` should be unique, otherwise non-unique values will be inserted as many times as the sum
        of their frequencies. Sometimes this situation is desirable.

        Also see the :meth:`add_items` and :meth:`add_iterator` methods.

        :param Iterable[object] items: An iterable of the items to insert into the instance.
        :param Callable[[object], int] mapping: A function that accepts an object and returns its frequency.
        :raises ValueError: If any frequency returned by `mapping` is less than 1.
        :return: The instance itself (self).
        :rtype: RandomSystematicPartitioning
        """
        for v in items:
            self.add_item(v, mapping(v))
        return self

    def add_iterator(self, items: Union[Iterator[object], Iterable[object]]) -> RandomSystematicPartitioning:
        """
        Add a stream of items into the instance.

        This method is equivalent to:

        .. code-block:: python

           for v in items:
               self.add_item(v, 1)
           return self

        See also the :meth:`add_item` and :meth:`add_items` methods.

        :param Union[Iterator[object], Iterable[object]] items: An iterator of the items to insert into the instance.
        :return: The instance itself (self).
        :rtype: RandomSystematicPartitioning
        """
        for v in items:
            self.add_item(v, 1)
        return self

    def reshuffle(self) -> RandomSystematicPartitioning:
        self.__rng.shuffle(self.__items)
        return self

    def partition(self) -> list[set[object]]:
        """
        Returns the partition held by this instance.

        More formally, this method returns a list of sets of elements, where each set contains :math:`k` unique
        elements. This method will raise :class:`ValueError` if the elements inserted up until the point that this
        method is called cannot be partitioned in such a way that both these conditions are met:

        #. The sum of the frequencies of all unique elements :math:`n` is divisible by :math:`k`.
        #. No element has frequency larger than :math:`n/k`.

        This method is deterministic and will always return the same result when executed consecutively without any
        insertion or reshuffling in-between. It always runs in time proportional to :math:`n` regardless of the
        properties of the elements inside the data structure.

        :raises ValueError: If the conditions specified (the divisibility and the max frequency) are not fulfilled.
        :return: The partition held in this instance.
        :rtype: list[set[object]]
        """
        if self.__n % self.__k != 0:
            raise ValueError("The number of elements acquired must be divisible by k")
        groups: list[set[object]] = []
        for i in range(self.__n // self.__k):
            groups.append(set())
        group_iterator: Iterator[set[object]] = itertools.cycle(groups)
        for x in self.__items:
            for j in range(self.__frequencies[x]):
                next_group: set[object] = next(group_iterator)
                if x in next_group:
                    raise ValueError(f"Element {x} was found too many times: {self.__frequencies[x]}")
                next_group.add(x)
        return groups

    def sample(self) -> set[object]:
        partitions: list[set[object]] = self.partition()
        return self.__rng.choice(partitions)
