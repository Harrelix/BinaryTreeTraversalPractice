from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from random import randint, choice
from typing import Generator, Callable

# max and min height of tree
# actually height of the tree will be more likely to be closer to MAX_HEIGHT
MAX_HEIGHT = 3
MIN_HEIGHT = 3

# max and min value of the nodes
# the random values will be equally distributed in the range [MIN_VAL, MAX_VAL]
MAX_VAL = 99
MIN_VAL = 0

# the width of each node when printing
NODE_WIDTH = len(str(MAX_VAL))

# traversal types to randomly choose from with equal probabilities
TRAVERSAL_TYPES = {"Preorder", "Inorder", "Postorder"}


class Tree:
    @dataclass
    class Node:
        val: int
        left: "Tree.Node" = None  # forward reference
        right: "Tree.Node" = None

        def PreorderTraversal(self) -> Generator[int, None, None]:
            yield self.val
            if not self.left is None:
                yield from self.left.PreorderTraversal()
            if not self.right is None:
                yield from self.right.PreorderTraversal()

        def InorderTraversal(self) -> Generator[int, None, None]:
            if not self.left is None:
                yield from self.left.InorderTraversal()
            yield self.val
            if not self.right is None:
                yield from self.right.InorderTraversal()

        def PostorderTraversal(self) -> Generator[int, None, None]:
            if not self.left is None:
                yield from self.left.PostorderTraversal()
            if not self.right is None:
                yield from self.right.PostorderTraversal()
            yield self.val

    class UnknownTraversalType(Exception):
        pass

    @classmethod
    def get_traversal_types(
        cls,
    ) -> dict[str, Callable[["Tree"], Generator[int, None, None]]]:
        """Return a dict that maps names of traversal types to the functions"""
        return {
            "Preorder": cls.PreorderTraversal,
            "Inorder": cls.InorderTraversal,
            "Postorder": cls.PostorderTraversal,
            "DFS": cls.DepthFirstTraversal,  # identical to preorder
            "BFS": cls.BreadthFirstTraversal,
        }

    def __init__(self):
        self.root = None
        self.height = -1

    def add(self, val: int) -> None:
        """Add val to tree"""
        # check if root is none
        if self.root is None:
            self.root = Tree.Node(val)
            self.height = 0
            return

        # traverse through the tree, keeping track of the height
        current_node = self.root
        current_height = 0
        while True:
            current_height += 1
            if val <= current_node.val:
                if current_node.left is None:
                    current_node.left = Tree.Node(val)
                    break
                else:
                    current_node = current_node.left
            else:
                if current_node.right is None:
                    current_node.right = Tree.Node(val)
                    break
                else:
                    current_node = current_node.right

        # update height
        self.height = max(self.height, current_height)

    def PreorderTraversal(self) -> Generator[int, None, None]:
        if self.root is None:
            return
        yield from self.root.PreorderTraversal()

    def InorderTraversal(self) -> Generator[int, None, None]:
        if self.root is None:
            return
        yield from self.root.InorderTraversal()

    def PostorderTraversal(self) -> Generator[int, None, None]:
        if self.root is None:
            return
        yield from self.root.PostorderTraversal()

    def BreadthFirstTraversal(self) -> Generator[int, None, None]:
        q = deque(maxlen=2 ** (self.height + 1))
        q.append(self.root)
        while not q:  # while q is not empty
            n = q.popleft()
            yield n.val
            if not n.left is None:
                q.append(n.left)
            if not n.right is None:
                q.append(n.right)

    def DepthFirstTraversal(self) -> Generator[int, None, None]:
        yield from self.BreadthFirstTraversal()

    def traverse(self, traverse_type: str) -> Generator[int, None, None]:
        d = Tree.get_traversal_types()
        if traverse_type in d:
            yield from d[traverse_type](self)
        else:
            raise Tree.UnknownTraversalType

    def print_tree(self) -> None:
        # Get the node at every layer
        q = deque(maxlen=2 ** (self.height + 1))
        q.append(self.root)
        layers = []
        for l in range(self.height + 1):
            layer = []
            for _ in range(2 ** l):
                n = q.popleft()
                if not n is None:
                    layer.append(n.val)
                    q.append(n.left)
                    q.append(n.right)
                else:
                    layer.append(None)
                    q.append(None)
                    q.append(None)
            layers.append(layer)

        # build the string to print
        tree_string = []
        spacing = 1  # space between each node, which are different from layer to layer
        indent = 0  # indent of the first node, which are different from layer to layer
        for layer in layers[::-1]:
            # traversing from the "bottom" up, because that's easier to calculate indents and spacings
            # add the nodes with indent and spacing
            layer_string = " " * indent + (" " * spacing).join(map(just_val, layer))
            tree_string.append(layer_string)

            # building the connectors
            if len(layer) == 1:  # don't need connectors if we're at the root layer
                break
            for i in range(spacing // 2 + 1):
                # add the indent
                connector_layer_string = " " * indent
                # indicate if n is left or right child, will alternates in the for loop
                isLeft = True
                for n in layer:
                    if n is None:  # don't need connectors if there're no child
                        connector_layer_string += " " * (NODE_WIDTH + i)
                    elif isLeft:
                        connector_layer_string += " " * (NODE_WIDTH - 1 + i) + "/"
                    else:
                        connector_layer_string += "\\" + " " * (NODE_WIDTH - 1 + i)

                    if isLeft:
                        connector_layer_string += " " * (spacing - (i * 2))
                    else:
                        connector_layer_string += " " * spacing

                    isLeft = not isLeft

                tree_string.append(connector_layer_string)

            # update indent and spacing
            indent = indent + (NODE_WIDTH + spacing) // 2
            spacing = NODE_WIDTH + 2 * spacing

        # since we built the tree_string in reverse, we have to reverse it before printing
        print("\n".join(tree_string[::-1]))


def generate_random_tree() -> Tree:
    """Generate a random tree with height and values between the bounds"""
    # determine the number of nodes
    max_num_node = 2 ** (MAX_HEIGHT + 1) - 1
    min_num_node = MIN_HEIGHT + 1
    num_node = randint(min_num_node, max_num_node)

    prev_tree = None

    current_tree = Tree()
    for _ in range(num_node):
        current_tree.add(randint(MIN_VAL, MAX_VAL))
        # stop adding node if exceeded MAX_HEIGHT
        if current_tree.height > MAX_HEIGHT:
            # return the tree prior to adding the last node
            return prev_tree
        prev_tree = deepcopy(current_tree)

    # return current tree with num_node if didn't exceed MAX_HEIGHT
    return current_tree


def just_val(v: int, l: int = len(str(MAX_VAL))) -> str:
    """Returns string of v that's right-justified"""
    if v is None:
        return " " * l
    return str(v).rjust(l)


def yes_no_prompt(msg: str) -> bool:
    """Prompt the user msg, an returns whether they answered yes or no"""
    while True:
        answer = input(msg + " ([y]/n)? ").lower()
        if answer == "" or answer == "y" or answer == "yes":
            return True
        elif answer == "n" or answer == "no":
            return False
        else:
            print("INVALID CHOICE:", answer)


def main():
    quitted = False
    while not quitted:
        # generate random tree
        tree = generate_random_tree()

        # prints the tree
        print("TREE:")
        tree.print_tree()

        # pick a random traversal type
        traverse_type = choice(tuple(TRAVERSAL_TYPES))
        traverse_order = list(tree.traverse(traverse_type))
        while True:
            user_inputs = input(
                "WRITE THE ORDER OF THE NODES WHEN TRAVERSING THE TREE IN "
                + traverse_type.upper()
                + ": "
            )
            try:
                user_inputs = list(map(int, user_inputs.split()))
            except ValueError:
                print("CAN'T PARSE INPUTS, TRY AGAIN!")
                continue

            for inp, val in zip_longest(user_inputs, traverse_order, fillvalue=0.1):
                # fill value is 0.1 because nothing in
                # user inputs and traverse_order is a float
                if inp != val:
                    try_again = yes_no_prompt("WRONG ANSWER, TRY AGAIN")
                    break
            else:
                print("THAT'S THE CORRECT ANSWER")
                again = yes_no_prompt("DO ANOTHER ONE")
                if not again:
                    quitted = True
                break

            if not try_again:
                # give user the correct answer
                print("THE CORRECT ANSWER IS:", " ".join(map(str, traverse_order)))
                again = yes_no_prompt("DO ANOTHER ONE")
                if not again:
                    quitted = True
                break


def validate_constants():
    assert MAX_HEIGHT >= MIN_HEIGHT, "MAX_HEIGHT smaller for MIN_HEIGHT"
    assert MIN_HEIGHT >= 1, "MIN_HEIGHT needs to be more than 1"
    assert MAX_VAL >= MIN_VAL, "MAX_VAL smaller than MIN_VAL"


if __name__ == "__main__":
    validate_constants()
    main()
