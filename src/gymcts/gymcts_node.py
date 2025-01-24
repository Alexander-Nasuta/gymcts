import uuid
import random
import math

from typing import TypeVar, Any, SupportsFloat, Callable, Generator

from gymcts.gymcts_gym_env import SoloMCTSGymEnv

from gymcts.logger import log

TSoloMCTSNode = TypeVar("TSoloMCTSNode", bound="SoloMCTSNode")




class SoloMCTSNode:

    # static properties
    best_action_weight: float = 0.05
    ubc_c  = 0.707


    # attributes
    visit_count: int = 0
    mean_value: float = 0
    max_value: float = -float("inf")
    min_value: float = +float("inf")
    terminal: bool = False
    state: Any


    def __str__(self, colored=False, action_space_n=None) -> str:
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


    def traverse_nodes(self) -> Generator[TSoloMCTSNode, None, None]:
        yield self
        if self.children:
            for child in self.children.values():
                yield from child.traverse_nodes()

    def get_root(self) -> TSoloMCTSNode:
        if self.is_root():
            return self
        return self.parent.get_root()

    def max_tree_depth(self):
        if self.is_leaf():
            return 0
        return 1 + max(child.max_tree_depth() for child in self.children.values())

    def n_children_recursively(self):
        if self.is_leaf():
            return 0
        return len(self.children) + sum(child.n_children_recursively() for child in self.children.values())



    def __init__(self,
                 action: int | None,
                 parent: TSoloMCTSNode | None,
                 env_reference: SoloMCTSGymEnv,
                 ):

        # field depending on whether the node is a root node or not
        self.action: int | None

        self.env_reference: SoloMCTSGymEnv
        self.parent: SoloMCTSNode | None
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
        #log.debug(f"saving state of node '{str(self)}' to memory location: {hex(id(self.state))}")
        self.visit_count: int = 0

        self.mean_value: float = 0
        self.max_value: float = -float("inf")
        self.min_value: float = +float("inf")

        # safe valid action instead of calling the environment
        # this reduces the compute but increases the memory usage
        self.valid_actions: list[int] = env_reference.get_valid_actions()
        self.children: dict[int, SoloMCTSNode] | None = None  # may be expanded later


    def reset(self) -> None:
        self.parent = None
        self.visit_count: int = 0

        self.mean_value: float = 0
        self.max_value: float = -float("inf")
        self.min_value: float = +float("inf")
        self.children: dict[int, SoloMCTSNode] | None = None  # may be expanded later

        # just setting the children of the parent node to None should be enough to trigger garbage collection
        # however, we also set the parent to None to make sure that the parent is not referenced anymore
        if self.parent:
            self.parent.reset()



    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return self.children is None or len(self.children) == 0

    def get_random_child(self) -> TSoloMCTSNode:
        if self.is_leaf():
            raise ValueError("cannot get random child of leaf node") #todo: maybe return self instead?

        return list(self.children.values())[random.randint(0, len(self.children) - 1)]

    def get_best_action(self) -> int:
        return max(self.children.values(), key=lambda child: child.get_score()).action

    def get_score(self) -> float: # todo: make it an attribute?
        # return self.mean_value
        assert 0 <= SoloMCTSNode.best_action_weight <= 1
        a = SoloMCTSNode.best_action_weight
        return (1-a) * self.mean_value + a * self.max_value

    def get_mean_value(self) -> float:
        return self.mean_value

    def get_max_value(self) -> float:
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
        c = SoloMCTSNode.ubc_c
        if self.visit_count == 0:
            return float("inf")
        return self.mean_value + c * math.sqrt(math.log(self.parent.visit_count) / (self.visit_count))