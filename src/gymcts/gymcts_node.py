import uuid
import random
import math

from typing import TypeVar, Any, SupportsFloat, Callable, Generator

from gymcts.gymcts_env_abc import GymctsABC

from gymcts.logger import log

TGymctsNode = TypeVar("TGymctsNode", bound="GymctsNode")


class GymctsNode:
    # static properties
    best_action_weight: float = 0.05 # weight for the best action
    ubc_c = 0.707 # exploration coefficient



    # attributes
    #
    # Note these attributes are not static. Their defined here to give developers a hint what fields are available
    # in the class. They are not static because they are not shared between instances of the class in scope of
    # this library.
    visit_count: int = 0 # number of times the node has been visited
    mean_value: float = 0 # mean value of the node
    max_value: float = -float("inf") # maximum value of the node
    min_value: float = +float("inf") # minimum value of the node
    terminal: bool = False # whether the node is terminal or not
    state: Any = None # state of the node

    def __str__(self, colored=False, action_space_n=None) -> str:
        """
        Returns a string representation of the node. The string representation is used for visualisation purposes.
        It is used for example in the mcts tree visualisation functionality.

        :param colored: true if the string representation should be colored, false otherwise. (ture is used by the mcts tree visualisation)
        :param action_space_n: the number of actions in the action space. This is used for coloring the action in the string representation.
        :return: a potentially colored string representation of the node.
        """
        if not colored:

            if not self.is_root():
                return f"(a={self.action}, N={self.visit_count}, Q_v={self.mean_value:.2f}, best={self.max_value:.2f}, ubc={self.ucb_score():.2f})"
            else:
                return f"(N={self.visit_count}, Q_v={self.mean_value:.2f}, best={self.max_value:.2f}) [root]"

        import gymcts.colorful_console_utils as ccu

        if self.is_root():
            return f"({ccu.CYELLOW}N{ccu.CEND}={self.visit_count}, {ccu.CYELLOW}Q_v{ccu.CEND}={self.mean_value:.2f}, {ccu.CYELLOW}best{ccu.CEND}={self.max_value:.2f})"

        if action_space_n is None:
            raise ValueError("action_space_n must be provided if colored is True")

        p = ccu.CYELLOW
        e = ccu.CEND
        v = ccu.CCYAN

        def colorful_value(value: float | int | None) -> str:
            if value == None:
                return f"{ccu.CGREY}None{e}"
            color = ccu.CCYAN
            if value == 0:
                color = ccu.CRED
            if value == float("inf"):
                color = ccu.CGREY
            if value == -float("inf"):
                color = ccu.CGREY

            if isinstance(value, float):
                return f"{color}{value:.2f}{e}"

            if isinstance(value, int):
                return f"{color}{value}{e}"

        root_node = self.get_root()
        mean_val = f"{self.mean_value:.2f}"

        return ((f"("
                 f"{p}a{e}={ccu.wrap_evenly_spaced_color(s=self.action, n_of_item=self.action, n_classes=action_space_n)}, "
                 f"{p}N{e}={colorful_value(self.visit_count)}, "
                 f"{p}Q_v{e}={ccu.wrap_with_color_scale(s=mean_val, value=self.mean_value, min_val=root_node.min_value, max_val=root_node.max_value)}, "
                 f"{p}best{e}={colorful_value(self.max_value)}") +
                (f", {p}ubc{e}={colorful_value(self.ucb_score())})" if not self.is_root() else ")"))

    def traverse_nodes(self) -> Generator[TGymctsNode, None, None]:
        """
        Traverse the tree and yield all nodes in the tree.

        :return: a generator that yields all nodes in the tree.
        """
        yield self
        if self.children:
            for child in self.children.values():
                yield from child.traverse_nodes()

    def get_root(self) -> TGymctsNode:
        """
        Returns the root node of the tree. The root node is the node that has no parent.

        :return: the root node of the tree.
        """
        if self.is_root():
            return self
        return self.parent.get_root()

    def max_tree_depth(self):
        """
        Returns the maximum depth of the tree. The depth of a node is the number of edges from
        the node to the root node.

        :return: the maximum depth of the tree.
        """
        if self.is_leaf():
            return 0
        return 1 + max(child.max_tree_depth() for child in self.children.values())

    def n_children_recursively(self):
        """
        Returns the number of children of the node recursively. The number of children of a node is the number of
        children of the node plus the number of children of all children of the node.

        :return: the number of children of the node recursively.
        """
        if self.is_leaf():
            return 0
        return len(self.children) + sum(child.n_children_recursively() for child in self.children.values())

    def __init__(self,
                 action: int | None,
                 parent: TGymctsNode | None,
                 env_reference: GymctsABC,
                 ):
        """
        Initializes the node. The node is initialized with the state of the environment and the action that was taken to
        reach the node. The node is also initialized with the parent node and the environment reference.

        :param action: the action that was taken to reach the node. If the node is a root node, this parameter is None.
        :param parent: the parent node of the node. If the node is a root node, this parameter is None.
        :param env_reference: a reference to the environment. The environment is used to get the state of the node and the valid actions.
        """

        # field depending on whether the node is a root node or not
        self.action: int | None

        self.env_reference: GymctsABC
        self.parent: GymctsNode | None
        self.uuid = uuid.uuid4()

        if parent is None:
            self.action = None
            self.parent = None
            if env_reference.is_terminal():
                raise ValueError("Root nodes shall not be terminal.")
        else:
            if action is None:
                raise ValueError("action must be provided if parent is not None")

            self.action = action
            self.parent = parent  # not None

        # fields that are always initialized the same way
        self.terminal: bool = env_reference.is_terminal()

        from copy import copy
        self.state = env_reference.get_state()
        # log.debug(f"saving state of node '{str(self)}' to memory location: {hex(id(self.state))}")
        self.visit_count: int = 0

        self.mean_value: float = 0
        self.max_value: float = -float("inf")
        self.min_value: float = +float("inf")

        # safe valid action instead of calling the environment
        # this reduces the compute but increases the memory usage
        self.valid_actions: list[int] = env_reference.get_valid_actions()
        self.children: dict[int, GymctsNode] | None = None  # may be expanded later

    def reset(self) -> None:
        self.parent = None
        self.visit_count: int = 0

        self.mean_value: float = 0
        self.max_value: float = -float("inf")
        self.min_value: float = +float("inf")
        self.children: dict[int, GymctsNode] | None = None  # may be expanded later

        # just setting the children of the parent node to None should be enough to trigger garbage collection
        # however, we also set the parent to None to make sure that the parent is not referenced anymore
        if self.parent:
            self.parent.reset()

    def is_root(self) -> bool:
        """
        Returns true if the node is a root node. A root node is a node that has no parent.

        :return: true if the node is a root node, false otherwise.
        """
        return self.parent is None

    def is_leaf(self) -> bool:
        """
        Returns true if the node is a leaf node. A leaf node is a node that has no children. A leaf node is a node that has no children.

        :return: true if the node is a leaf node, false otherwise.
        """
        return self.children is None or len(self.children) == 0

    def get_random_child(self) -> TGymctsNode:
        """
        Returns a random child of the node. A random child is a child that is selected randomly from the list of children.
        :return:
        """
        if self.is_leaf():
            raise ValueError("cannot get random child of leaf node")  # todo: maybe return self instead?

        return list(self.children.values())[random.randint(0, len(self.children) - 1)]

    def get_best_action(self) -> int:
        """
        Returns the best action of the node. The best action is the action that has the highest score.
        The score is calculated using the get_score() method. The best action is the action that has the highest score.
        The best action is the action that has the highest score.

        :return: the best action of the node.
        """
        return max(self.children.values(), key=lambda child: child.get_score()).action

    def get_score(self) -> float:  # todo: make it an attribute?
        """
        Returns the score of the node. The score is calculated using the mean value and the maximum value of the node.
        The score is calculated using the formula: score = (1 - a) * mean_value + a * max_value
        where a is the best action weight.

        :return: the score of the node.
        """
        # return self.mean_value
        assert 0 <= GymctsNode.best_action_weight <= 1
        a = GymctsNode.best_action_weight
        return (1 - a) * self.mean_value + a * self.max_value

    def get_mean_value(self) -> float:
        return self.mean_value

    def get_max_value(self) -> float:
        """
        Returns the maximum value of the node. The maximum value is the maximum value of the node.

        :return: the maximum value of the node.
        """
        return self.max_value

    def ucb_score(self):
        """
        The score for an action that would transition between the parent and child.
        prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)

        if child.visit_count > 0:
            # The value of the child is from the perspective of the opposing player
            value_score = -child.value()
        else:
            value_score = 0

            return value_score + prior_score

        :return:
        """
        if self.is_root():
            raise ValueError("ucb_score can only be called on non-root nodes")
        # c = 0.707 # todo: make it an attribute?
        c = GymctsNode.ubc_c
        if self.visit_count == 0:
            return float("inf")
        return self.mean_value + c * math.sqrt(math.log(self.parent.visit_count) / (self.visit_count))
