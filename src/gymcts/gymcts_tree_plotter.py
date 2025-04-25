from typing import Any, Generator

from gymcts.gymcts_node import GymctsNode

from gymcts.logger import log


def _generate_mcts_tree(
        start_node: GymctsNode = None,
        prefix: str = None,
        depth: int = None,
        exclude_unvisited_nodes_from_render: bool = True,
        action_space_n: int = None
) -> Generator[str, Any | None, None]:
    """
    Generates a tree representation of the MCTS tree starting from the given node.

    This is a recursive function that generates a tree representation of the MCTS tree starting from the given node. The

    :param start_node: the node to start from
    :param prefix: used to format the tree
    :param depth: used to limit the depth of the tree
    :param exclude_unvisited_nodes_from_render: used to exclude unvisited nodes from the render
    :param action_space_n: the number of actions in the action space
    :return: a list of strings representing the tree
    """
    if prefix is None:
        prefix = ""
    import gymcts.colorful_console_utils as ccu

    if start_node is None:
        raise ValueError("start_node must not be None")

    if action_space_n is None:
        log.warning("action_space_n is None, defaulting to 100")
        action_space_n = 100

    # prefix components:
    space = '    '
    branch = '│   '
    # pointers:
    tee = '├── '
    last = '└── '

    contents = start_node.children.values() if start_node.children is not None else []
    if exclude_unvisited_nodes_from_render:
        contents = [node for node in contents if node.visit_count > 0]
    # contents each get pointers that are ├── with a final └── :
    # pointers = [tee] * (len(contents) - 1) + [last]
    pointers = [tee for _ in range(len(contents) - 1)] + [last]

    for pointer, current_node in zip(pointers, contents):
        n_item = current_node.parent.action if current_node.parent is not None else 0
        n_classes = action_space_n

        pointer = ccu.wrap_evenly_spaced_color(
            s=pointer,
            n_of_item=n_item,
            n_classes=n_classes,
        )

        yield prefix + pointer + f"{current_node.__str__(colored=True, action_space_n=n_classes)}"
        if current_node.children and len(current_node.children):  # extend the prefix and recurse:
            # extension = branch if pointer == tee else space
            extension = branch if tee in pointer else space
            # i.e. space because last, └── , above so no more |
            extension = ccu.wrap_evenly_spaced_color(
                s=extension,
                n_of_item=n_item,
                n_classes=n_classes,
            )
            if depth is not None and depth <= 0:
                continue
            yield from _generate_mcts_tree(
                current_node,
                prefix=prefix + extension,
                action_space_n=action_space_n,
                depth=depth - 1 if depth is not None else None
            )


def show_mcts_tree(
        start_node: GymctsNode = None,
        tree_max_depth: int = None,
        action_space_n: int = None
) -> None:
    """
    Renders the MCTS tree starting from the given node.

    :param start_node: the node to start from
    :param tree_max_depth: the maximum depth of the tree to render
    :param action_space_n: the number of actions in the action space
    """
    print(start_node.__str__(colored=True, action_space_n=action_space_n))
    for line in _generate_mcts_tree(start_node=start_node, depth=tree_max_depth):
        print(line)
