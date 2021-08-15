"""
Implementations of the SE-A, SE-B and SE-C graph generators and some auxiliary utilities.
"""
from __future__ import annotations

import typing
from random import Random

import networkx as nx

T = typing.TypeVar('T')


def se_a(n: int, rng: Random) -> nx.Graph:
    """
    Returns a random graph using the SE-A preferential attachment algorithm.

    This algorithm generates an undirected, unweighted graph, without self loops or multiple edges.

    :param n: Number of nodes of the final graph.
    :type n: int

    :param rng: Random number generator.
    :type rng: Random

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


def se_b(n: int, m: int, rng: Random, initial_graph: nx.Graph = None) -> nx.Graph:
    """
    Returns a random graph using the SE-B preferential attachment algorithm.

    This algorithm generates an undirected, unweighted graph, without self loops or multiple edges.

    :param n: Number of nodes of the final graph.
    :type n: int

    :param m: Number of edges to attach from a new node to existing nodes.
    :type m: int

    :param rng: Random number generator.
    :type rng: Random

    :param initial_graph: Initial network for the algorithm. It must be an undirected graph without self loops of
        multiple edges. The initial graph must satisfy the divisibility `2|E_0|/m` and no vertex can have degree higher
        than `2|E_0|/m`. It should be connected, although this is not enforced. The initial graph will be copied before
        being used. This argument is optional and if `None` is given then the process starts from a complete graph of
        `m` nodes and `m(m-1)` edges.
    :type initial_graph: Graph, optional

    :raises ValueError: If `n >= |V_0| >= m >= 2` is not satisfied or if the conditions required for the `initial_graph`
        are not fulfilled, namely the divisibility `2|E_0|/m` and the maximum degree of `2|E_0|/m`.

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
        # Create two new hyperedges and add the source on both. Here we also create an alternating iterator for
        # convenience.
        hyperedge_x: set[object] = {source}
        hyperedge_y: set[object] = {source}
        new_hyperedges: typing.Iterator[set[object]] = rotating_iterator([hyperedge_x, hyperedge_y])
        # Select a random hyperedge and add its elements into the new hyperedges using a random split. At the same time
        # add the new edges in the graph.
        for v in shuffled(rng.choice(hyperedge_list), rng):
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
        # Add the new hyperedges on the hyperedge list
        hyperedge_list.append(next(new_hyperedges))
        hyperedge_list.append(next(new_hyperedges))
        # Advance source
        source += 1

    # Return the graph and we're done
    return g


def se_c(n: int, m: int, rng: Random, initial_graph: nx.Graph = None) -> nx.Graph:
    """
    Returns a random graph using the SE-C preferential attachment algorithm.

    This algorithm generates an undirected, unweighted graph, without self loops or multiple edges.

    :param n: Number of nodes of the final graph.
    :type n: int

    :param m: Number of edges to attach from a new node to existing nodes.
    :type m: int

    :param rng: Random number generator.
    :type rng: Random

    :param initial_graph: Initial network for the algorithm. It must be an undirected graph without self loops of
        multiple edges. The initial graph must satisfy the divisibility `2|E_0|/m` and no vertex can have degree higher
        than `2|E_0|/m`. It should be connected, although this is not enforced. The initial graph will be copied before
        being used. This argument is optional and if `None` is given then the process starts from a complete graph of
        `m` nodes and `m(m-1)` edges.
    :type initial_graph: Graph, optional

    :raises ValueError: If `n >= |V_0| >= m >= 2` is not satisfied or if the conditions required for the `initial_graph`
        are not fulfilled, namely the divisibility `2|E_0|/m` and the maximum degree of `2|E_0|/m`.

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
        rst: RandomSystematicPartitioning = RandomSystematicPartitioning(m, rng)
        # Select one random old hyperedge and insert its elements into the RST. At the same time create the edges too.
        for v in shuffled(rng.choice(hyperedge_list), rng):
            rst.add_item(v, 1)
            g.add_edge(source, v)
        # Insert m copies of source into the RST
        rst.add_item(source, m)
        # Randomly select m-2 old hyperedges and add them into the RST
        random_old_hyperedges: list[int] = list(random_selections(len(hyperedge_list), m - 2, rng))
        for i in random_old_hyperedges:
            for v in hyperedge_list[i]:
                rst.add_item(v, 1)
        # Run the partitioning in the RST
        rst_iter: typing.Iterator[set[object]] = iter(rst.partition())
        # Replace m-2 rows of the RST into the random m-2 old hyperedges previously selected and insert the other 2
        for i in random_old_hyperedges:
            hyperedge_list[i] = next(rst_iter)
        hyperedge_list.append(next(rst_iter))
        hyperedge_list.append(next(rst_iter))
        # Advance source
        source += 1

    # Return the graph and we're done
    return g


def rotating_iterator(a: typing.Iterable[T]) -> typing.Iterator[T]:
    """
    Returns a rotating iterator over the elements of the iterable `a`.

    A rotating iterator traverses the elements of `a` once and then starts over. The rotating iterator is infinite and
    never stops. For example the snippet

    .. code-block:: python

       ri: Iterator[int] = rotating_iterator([1, 2])
       for i in range(10):
           print(next(ri))

    will alternate between the values 1 and 2 for 5 times before terminating.

    :param a: The input iterable.
    :type a: Iterable[T]

    :return: A rotating iterator over the elements of `a`.
    :rtype: Iterator[T]
    """

    while True:
        for e in a:
            yield e


def shuffled(a: typing.Iterable[T], rng: Random) -> typing.Iterator[T]:
    """
    Returns a generator that iterates through the values of the input iterable in random order.

    This function will make a copy of the input iterable before using it. It is useful when the input is not a random
    access collection, for example a set, and you don't necessarily need to have the items in a concrete collection, for
    example you wish to perform a one-pass operation on them, for example you are using it as intermediate storage. In
    other cases, it probably won't be any good in terms of performance.

    :param a: The input iterable.
    :type a: Iterable[object]

    :param rng: The random number generator.
    :type rng: Random

    :return: A generator for the values of the input collection in random order.
    :rtype: Iterator[object]
    """

    a: list[object] = list(a)
    for i in range(0, len(a) - 1):
        j: int = rng.randrange(i, len(a))
        yield a[j]
        a[j] = a[i]
    yield a[-1]


def random_selections(n: int, k: int, rng: Random) -> typing.Iterator[int]:
    """
    Performs an unweighted selection without replacement of `k` elements from a population of `n` elements.

    The population and the sample are represented by their indices and, as a result, this method will return
    `k` random and discrete indices in the range `[0,n)`. The selection is performed in such a way that the higher order
    inclusion probabilities of all `k`-tuples are equal. In practice, this means that if `k = 2`, all pairs of numbers
    are equally likely to appear as the result of this method. The operation of this algorithm is based on a virtual
    shuffling of the array `[0,n)` where the first `k` elements are then being returned efficiently.

    If you need to reuse the result of this operation you need to store it in a collection. A :class:`set` would be a
    convenient container as the elements returned are unique and don't have any particular order at which they are
    returned:

    .. code-block:: python

       random_numbers = set(random_selections(10, 3, Random()))

    This method returns a generator that will be fully consumed in time proportional to `k` in the worst case and is not
    prone to rejections due to the selection of duplicate elements. The generator uses memory proportional to `k` in the
    worst case.

    :param n: The size of the population.
    :type n: int

    :param k: The size of the sample.
    :type k: int

    :param rng: The random number generator to use.
    :type rng: Random

    :raises ValueError: If the condition `n >= k >= 0` is not satisfied.

    :return: A generator that holds the values of `k` random and discrete integers in the range `[0,n)`.
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


class RandomSystematicPartitioning:
    """
    Implementation of the random systematic partitioning scheme.

    Random systematic partitioning (RST) randomly partitions the input collection into groups of `k` elements such that
    each group contains only unique elements. The number of groups itself is implicitly defined by the ratio of the sum
    of frequencies over `k`. The algorithm implemented in this class is heavily influenced by the systematic random
    sampling design. If the elements cannot be partitioned in such a way, then :class:`ValueError` is raised from the
    :meth:`~se.RandomSystematicPartitioning.partition` method.

    The interface allows you to add elements into the RST using the methods
    :meth:`~se.RandomSystematicPartitioning.add_item` and :meth:`~se.RandomSystematicPartitioning.add_items` and then it
    is possible to use the method :meth:`~se.RandomSystematicPartitioning.partition` to do the actual partitioning. By
    itself the :meth:`~se.RandomSystematicPartitioning.partition` does not change the state of the instance.

    Below is a typical use case that will partition 2 copies of the element 'a' and the elements 'b' and 'c' into groups
    of 2.

    .. code-block:: python

       rst = RandomSystematicPartitioning(2, random.Random())
       rst.add_item('a', 2)
       rst.add_item('b', 1)
       rst.add_item('c', 1)
       print(rst.partition())

    Because the element 'a' exists twice in the input collection, the only possible partitioning is the two groups
    (a, b) and (a, c). The actual order of the groups returned is not important as the elements are randomly shuffled
    anyway. In fact, the probability of receiving any of the two partitions is 50%. Keep in mind that the
    :meth:`~se.RandomSystematicPartitioning.partition` method is deterministic and will return the same result on
    consecutive calls. The actual randomization (or shuffling) is performed online as elements enter the instance.
    Statistically independent partitions (even of the same input data) must rely on different instances of this class
    and not different invocations of the :meth:`~se.RandomSystematicPartitioning.partition` method. As a result, calling
    :meth:`~se.RandomSystematicPartitioning.partition` is normally not required or needed.

    The :meth:`~se.RandomSystematicPartitioning.add_items` may be used to insert items into the RST when the population
    is known and exists in a concrete collection and at the same time a mapping function exists that maps the elements
    with their frequencies. In a simple case where the elements along with their frequencies are stored in a dictionary
    `dict[object, int]` a typical use case would be the following:

    .. code-block:: python

       frequencies = {'a': 2, 'b': 1, 'c': 1}
       rst = RandomSystematicPartitioning(2, random.Random())
       rst.add_items(frequencies.keys(), lambda x: frequencies[x])
       print(rst.partition())

    The interface of this class also allows the use of builder-like syntax because the `add_*` methods return the
    instance itself, for example:

    .. code-block:: python

       rst = RandomSystematicPartitioning(2, random.Random())
       rst.add_item('a', 2)
          .add_item('b', 1)
          .add_item('c', 1)
       print(rst.partition())
    """

    def __init__(self, k: int, rng: Random) -> None:
        """
        Construct an instance of this class with a group size `k` and a random number generator `rng`.

        The constructor runs in constant time.

        :param k: The size of each group.
        :type k: int

        :param rng: The random number generator.
        :type rng: Random

        :raises ValueError: If `k < 1`.
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

        Also see the :meth:`add_items` method.

        :param item: The item to insert.
        :type item: object

        :param frequency: The frequency of the item.
        :type frequency: int

        :raises ValueError: If `frequency < 1`.

        :return: The instance itself (self).
        :type: RandomSystematicPartitioning
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

    def add_items(self, items: typing.Iterable[object],
                  mapping: typing.Callable[[object], int]) -> RandomSystematicPartitioning:
        """
        Adds a collection of items along with their frequencies into the instance.

        This method is equivalent to:

        .. code-block:: python

           for v in items:
               self.add_item(v, mapping(v))
           return self

        The elements in `items` should be unique, otherwise non-unique values will be inserted as many times as the sum
        of their frequencies.

        Also see the :meth:`add_items` method.

        :param items: An iterable of the items to insert into the instance.
        :type items: Iterable[object]

        :param mapping: A function that accepts an object and returns its frequency.
        :type mapping: Callable[[object], int]

        :raises ValueError: If any frequency returned by `mapping` is less than 1.

        :return: The instance itself (self).
        :type: RandomSystematicPartitioning
        """

        for v in items:
            self.add_item(v, mapping(v))
        return self

    def partition(self) -> list[set[object]]:
        """
        Returns the partition held by this instance.

        More formally, this method returns a list of sets of elements, where each set contains `k` unique elements. This
        method will raise :class:`ValueError` if the elements inserted up until the point that this method is called
        cannot be partitioned in such a way that both these conditions are met:

        #. The sum of the frequencies of all unique elements `n` is divisible by `k`.
        #. No element has frequency larger than `n/k`.

        This method is deterministic and will always return the same result when executed consecutively without any
        insertion in-between. It always runs in time proportional to `n` regardless of the properties of the elements
        inside the data structure.

        :raises ValueError: If the conditions specified (the divisibility and the max frequency) are not fulfilled.

        :return: The partition held in this instance.
        :rtype: list[set[object]]
        """

        if self.__n % self.__k != 0:
            raise ValueError("The number of elements acquired must be divisible by k")
        groups: list[set[object]] = []
        for i in range(self.__n // self.__k):
            groups.append(set())
        group_iterator: typing.Iterator[set[object]] = rotating_iterator(groups)
        for x in self.__items:
            for j in range(self.__frequencies[x]):
                next_group: set[object] = next(group_iterator)
                if x in next_group:
                    raise ValueError(f"Element {x} was found too many times: {self.__frequencies[x]}")
                next_group.add(x)
        return groups
