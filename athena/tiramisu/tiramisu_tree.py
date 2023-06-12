from typing import Dict, List, Tuple
from athena.tiramisu.tiramisu_iterator_node import IteratorNode
import itertools


class TiramisuTree:
    """This class represents the tree structure of a Tiramisu program.
    It is composed of a list of IteratorNode objects, each of which represents an iterator
    in the Tiramisu program. Each IteratorNode object contains information about its
    parent iterator, child iterators, lower and upper bounds, and the computations that
    it is associated with.

    Attributes:
    ----------
    `roots`: `List[str]`
        List of names of the root iterators in the Tiramisu program.
    `iterators`: `Dict[str, IteratorNode]`
        Dictionary of IteratorNode objects, indexed by the name of the iterator.
    `computations`: `List[str]`
        List of names of the computations in the Tiramisu program.

    Methods:
    -------
    `from_annotations(cls, annotations: Dict) -> "TiramisuTree":`
        Creates a TiramisuTree object from the annotations of a Tiramisu program.
    `get_candidate_sections(self) -> List[List[str]]:`
        Returns a list of candidate sections in the Tiramisu program.

    """

    def __init__(self) -> None:
        self.roots: List[str] = []
        self.iterators: Dict[str, IteratorNode] = {}
        self.computations: List[str] = []

    def add_root(self, root: str) -> None:
        self.roots.append(root)

    def add_computation(self, comp: str) -> None:
        self.computations.append(comp)

    @classmethod
    def from_annotations(cls, annotations: Dict) -> "TiramisuTree":
        """
        Creates a TiramisuTree object from the annotations of a Tiramisu program.

        Parameters:
        ----------
        `annotations`: `Dict`
            Annotations of a Tiramisu program.

        Returns:
        -------
        `tiramisu_space`: `TiramisuTree`
        """
        tiramisu_space = cls()

        iterators = annotations["iterators"]

        for iterator in iterators:
            parent_iterator = iterators[iterator]["parent_iterator"]
            iterator_level = None
            if parent_iterator is None:
                tiramisu_space.add_root(iterator)
                iterator_level = 0
            else:
                iterator_level = tiramisu_space.iterators[parent_iterator].level + 1

            tiramisu_space.iterators[iterator] = IteratorNode(
                name=iterator,
                lower_bound=iterators[iterator]["lower_bound"],
                upper_bound=iterators[iterator]["upper_bound"],
                child_iterators=iterators[iterator]["child_iterators"],
                computations_list=iterators[iterator]["computations_list"],
                parent_iterator=iterators[iterator]["parent_iterator"],
                level=iterator_level,
            )
            tiramisu_space.computations.extend(iterators[iterator]["computations_list"])

        return tiramisu_space

    def __str__(self) -> str:
        return f"Roots: {self.roots}\nComputations: {self.computations}\nIterators: {self.iterators}"

    def __repr__(self) -> str:
        return self.__str__()

    def get_candidate_sections(self) -> Dict[str, List[List[str]]]:
        """
        Returns a dictionary with lists of candidate sections for each root iterator.

        Returns:
        -------

        `candidate_sections`: `Dict[str, List[List[str]]]`
            Dictionary with lists of candidate sections for each root iterator.
        """

        candidate_sections = {}
        for root in self.roots:
            nodes_to_visit = [root]
            list_candidate_sections = []
            for node in nodes_to_visit:
                candidate_section, new_nodes_to_visit = self._get_section_of_node(node)
                list_candidate_sections.append(candidate_section)
                nodes_to_visit.extend(new_nodes_to_visit)
            candidate_sections[root] = list_candidate_sections
        return candidate_sections

    def _get_section_of_node(self, node_name: str) -> Tuple[List[str], List[str]]:
        candidate_section = [node_name]
        current_node = self.iterators[node_name]

        while (
            len(current_node.child_iterators) == 1
            and len(current_node.computations_list) == 0
        ):
            next_node_name = current_node.child_iterators[0]
            candidate_section.append(next_node_name)
            current_node = self.iterators[next_node_name]

        if current_node.child_iterators:
            return candidate_section, current_node.child_iterators
        return candidate_section, []

    def get_level_of_node(self, node_name: str) -> int:
        """
        Returns the level of the node in the program tree.

        Parameters:
        ----------
        `node_name`: `str`
            The name of the node.

        `program_tree`: `TiramisuTree`
            The Tiramisu tree of the program.

        Returns:
        -------
        int
            The level of the node.
        """

        node = self.iterators[node_name]
        level = 0

        while node.parent_iterator:
            level += 1
            node = self.iterators[node.parent_iterator]

        return level

    def interchange(self, node1: str, node2: str) -> None:
        """
        Interchanges the positions of two nodes in the program tree.

        Parameters:
        ----------
        `node1`: `str`
            The name of the first node.

        `node2`: `str`
            The name of the second node.
        """

        node1_parent = self.iterators[node1].parent_iterator
        node2_parent = self.iterators[node2].parent_iterator

        if node2_parent == node1:
            new_node1_parent = node2
        else:
            new_node1_parent = node2_parent

        new_node1 = IteratorNode(
            name=node1,
            parent_iterator=new_node1_parent,
            lower_bound=self.iterators[node1].lower_bound,
            upper_bound=self.iterators[node1].upper_bound,
            child_iterators=self.iterators[node2].child_iterators,
            computations_list=self.iterators[node2].computations_list,
            level=self.iterators[node2].level,
        )

        if node1_parent == node2:
            new_node2_parent = node1
        else:
            new_node2_parent = node1_parent

        new_node2 = IteratorNode(
            name=node2,
            parent_iterator=new_node2_parent,
            lower_bound=self.iterators[node2].lower_bound,
            upper_bound=self.iterators[node2].upper_bound,
            child_iterators=self.iterators[node1].child_iterators,
            computations_list=self.iterators[node1].computations_list,
            level=self.iterators[node1].level,
        )

        if node1_parent:
            if node1_parent == node2:
                new_node1.child_iterators = [
                    node2 if x == node1 else x for x in new_node1.child_iterators
                ]
            else:
                parent_node = self.iterators[node1_parent]  # type: ignore

                parent_node.child_iterators[
                    parent_node.child_iterators.index(node1)
                ] = node2
        else:
            self.roots[self.roots.index(node1)] = node2

        if node2_parent:
            if node2_parent == node1:
                new_node2.child_iterators = [
                    node1 if x == node2 else x for x in new_node2.child_iterators
                ]

            else:
                parent_node = self.iterators[node2_parent]  # type: ignore

                parent_node.child_iterators[
                    parent_node.child_iterators.index(node2)
                ] = node1
        else:
            self.roots[self.roots.index(node2)] = node1

        self.iterators[node1] = new_node1
        self.iterators[node2] = new_node2

    def get_candidate_computations(self, candidate_node_name: str) -> List[str]:
        """Get the list of computations impacted by this node

        Parameters:
        ----------
        `candidate_node`: `IteratorNode`
            The candidate node for parallelization.

        `program_tree`: `TiramisuTree`
            The Tiramisu tree of the program.

        Returns:
        -------
        `list`
            List of computations impacted by the node
        """

        computations: List[str] = []
        candidate_node = self.iterators[candidate_node_name]

        computations += candidate_node.computations_list

        for child in candidate_node.child_iterators:
            computations += self.get_candidate_computations(child)

        return computations