from typing import Callable, List
import operator


class SegmentTree:
    """Implementation of a Segment Tree data structure."""

    def __init__(
        self, size: int, operation: Callable[[object, object], object], identity: object
    ) -> None:
        """Implementation of a Segment Tree data structure.

        Args:
            size (int): the length of the array to represent (must be power of 2).
            operation (Callable[[object, object], object]): the operation (lambda) to answer quick queries for.
            identity (object): the identity element for said operation.
        """
        self.size = size
        self.operation = operation
        self.identity = identity
        self.data = [self.identity for _ in range(2 * self.size)]

    def _reduce(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> object:
        """Recursively find the result of the operation in a given range.

        Args:
            start (int): the index at the start of the range.
            end (int): the index at the end of the range.
            node (int): the index of the node the explorative call is at.
            node_start (int): the index at the start of the range the node is responsible for.
            node_end (int): the index at the end of the range the node is responsible for.

        Returns:
            object: the result of the operation at overlap of the given range and the node's range.
        """
        if start == node_start and end == node_end:  # Total overlap
            return self.data[node]

        node_mid = (node_start + node_end) // 2
        if end <= node_mid:  # Partial overlap, but exclusively left
            return self._reduce(start, end, 2 * node, node_start, node_mid)
        elif node_mid + 1 <= start:  # Partial overlap, but exclusively right
            return self._reduce(start, end, 2 * node + 1, node_mid + 1, node_end)
        else:  # Partial overlap, but overlap is split in both directions
            return self.operation(
                self._reduce(start, node_mid, 2 * node, node_start, node_mid),
                self._reduce(node_mid + 1, end, 2 * node + 1, node_mid + 1, node_end),
            )

    def reduce(self, start=0, end=None) -> object:
        """Tail recursively find the result of the operation in a given range.

        Args:
            start (int, optional): the index at the start of the range. Defaults to 0.
            end ([type], optional): the index at the end of the range. Defaults to None.

        Returns:
            object: the result of the operation at the given range.
        """
        if end is None:  # Argument not provided
            end = self.size
        if end < 0:  # Argument is relative to the end of the array
            end += self.size

        end -= 1  # Adjust to index
        return self._reduce(start, end, 1, 0, self.size - 1)

    def __setitem__(self, index: int, item: object):
        """Set an item on the given index.

        Args:
            index (int): the index of the item to set.
            item (object): the item to set.
        """
        index += self.size  # Get true index in the array
        self.data[index] = item
        index //= 2  # Navigate to parent node
        while index >= 1:  # Update all the way to the root node
            self.data[index] = self.operation(
                self.data[2 * index], self.data[2 * index + 1]
            )
            index //= 2

    def __getitem__(self, index: int) -> object:
        """Access an item on the given index.

        Args:
            index (int): the index of the item to access.

        Returns:
            object: the element at the given index.
        """
        return self.data[self.size + index]

    def get_values(self, end: int = None) -> List[object]:
        """Get the bottom-level leaf values of the segment tree.

        Args:
            end (int): the index of the last element to include

        Returns:
            List[object]: the saved values.
        """
        return self.data[self.size : self.size + end]


class SumSegmentTree(SegmentTree):
    """A Segment Tree that allows for efficient sum queries."""

    def __init__(self, size: int):
        """A Segment Tree that allows for efficient sum queries.

        Args:
            size (int): the length of the array to represent (must be power of 2).
        """
        super().__init__(size=size, operation=operator.add, identity=0.0)

    def sum(self, start=0, end=None) -> float:
        """Return the sum of the elements in the range

        Args:
            start (int, optional): the index of the first element in the sum. Defaults to 0.
            end ([type], optional): the index of the final element in the sum. Defaults to None.

        Returns:
            float: the sum in the given range
        """
        return super().reduce(start, end)


class MinSegmentTree(SegmentTree):
    """A Segment Tree that allows for efficient min queries."""

    def __init__(self, size: int):
        """A Segment Tree that allows for efficient min queries.

        Args:
            size (int): the length of the array.
        """
        super().__init__(size=size, operation=min, identity=float("inf"))

    def min(self, start=0, end=None) -> float:
        """Return the minimum of all the elements in the range

        Args:
            start (int, optional): the index of the first element to compare to. Defaults to 0.
            end ([type], optional): the index of the final element to compare to. Defaults to None.

        Returns:
            float: the minimum element in the tree
        """
        return super().reduce(start, end)
