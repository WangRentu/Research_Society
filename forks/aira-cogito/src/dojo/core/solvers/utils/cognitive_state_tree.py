# Copyright (c) 2026 Research Society
# MC-ESES: Monte Carlo Endogenous State Evolution Search
#
# CognitiveStateNode — a node in the cognitive state search tree.
# Unlike standard MCTS (which searches over code solutions), MC-ESES
# searches over cognitive states.  Each node holds a CognitiveState z_t
# and MCTS tracking (explore_count, node_value).  Code solutions are
# generated as projections E(z_t) and stored in the flat Journal.

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dojo.core.solvers.utils.cognitive_state import CognitiveState


# ---------------------------------------------------------------------------
# UCT helpers (duplicated from mcts.py to avoid cross-solver dependency)
# ---------------------------------------------------------------------------

def normalise_q_value(
    q_value: float, global_max_q_val: float, global_min_q_val: float
) -> float:
    if global_max_q_val == global_min_q_val:
        return 0.5
    return (q_value - global_min_q_val) / (global_max_q_val - global_min_q_val)


def uct_value(
    q_value: float,
    explore_count: int,
    parent_explore_count: int,
    uct_c: float,
    global_max_q_val: float,
    global_min_q_val: float,
    intrinsic_quality: float = 0.0,
    intrinsic_weight: float = 0.2,
) -> float:
    if explore_count == 0:
        return float("inf")  # prefer unvisited nodes
    norm_q = normalise_q_value(q_value, global_max_q_val, global_min_q_val)
    exploration = math.sqrt(math.log(parent_explore_count) / explore_count)
    # Combined value: extrinsic Q + intrinsic state quality
    combined_q = (1.0 - intrinsic_weight) * norm_q + intrinsic_weight * intrinsic_quality
    return combined_q + uct_c * exploration


# ---------------------------------------------------------------------------
# CognitiveStateNode
# ---------------------------------------------------------------------------

@dataclass
class CognitiveStateNode:
    """A node in the cognitive state search tree for MC-ESES.

    Each node holds a CognitiveState z_t and MCTS bookkeeping.
    The actual code/metric data lives in the Journal; this node
    only stores a pointer (``source_node_step``) to the Journal
    Node whose execution + reflection produced this cognitive state.
    """

    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cognitive_state: CognitiveState = field(default_factory=CognitiveState)
    parent: Optional["CognitiveStateNode"] = field(default=None, repr=False)
    children: List["CognitiveStateNode"] = field(default_factory=list)

    # MCTS tracking
    explore_count: int = 0
    node_value: float = 0.0

    # Multi-dimensional backprop tracking
    validity_count: int = 0      # number of non-buggy expansions in subtree
    total_expansions: int = 0    # total expansions in subtree (for validity rate)

    # Link back to the Journal Node that gave birth to this state
    source_node_step: Optional[int] = None

    # Tree metadata
    depth: int = 0
    ctime: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    # MCTS value helpers
    # ------------------------------------------------------------------

    def q_value(self, lower_is_better: bool = False) -> float:
        if self.explore_count == 0:
            return -1e8
        q = self.node_value / self.explore_count
        if lower_is_better:
            q = -q
        return q

    def add_value(self, value: float) -> None:
        if value is not None:
            self.node_value += value

    def increment_explore_count(self) -> None:
        self.explore_count += 1

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    # ------------------------------------------------------------------
    # Serialisation (recursive)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Recursively serialise the subtree rooted at this node."""
        return {
            "node_id": self.node_id,
            "cognitive_state": self.cognitive_state.to_dict(),
            "explore_count": self.explore_count,
            "node_value": self.node_value,
            "validity_count": self.validity_count,
            "total_expansions": self.total_expansions,
            "source_node_step": self.source_node_step,
            "depth": self.depth,
            "ctime": self.ctime,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CognitiveStateNode":
        """Recursively deserialise a subtree."""
        node = cls(
            node_id=d.get("node_id", uuid.uuid4().hex),
            cognitive_state=CognitiveState.from_dict(d.get("cognitive_state", {})),
            explore_count=d.get("explore_count", 0),
            node_value=d.get("node_value", 0.0),
            validity_count=d.get("validity_count", 0),
            total_expansions=d.get("total_expansions", 0),
            source_node_step=d.get("source_node_step"),
            depth=d.get("depth", 0),
            ctime=d.get("ctime", time.time()),
        )
        for child_dict in d.get("children", []):
            child = cls.from_dict(child_dict)
            child.parent = node
            node.children.append(child)
        return node


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def serialize_cs_tree(root: CognitiveStateNode) -> Dict[str, Any]:
    """Serialise the full cognitive state tree for checkpointing."""
    return root.to_dict()


def deserialize_cs_tree(data: Dict[str, Any]) -> CognitiveStateNode:
    """Deserialise a cognitive state tree from checkpoint data."""
    return CognitiveStateNode.from_dict(data)


def tree_stats(root: CognitiveStateNode) -> Dict[str, Any]:
    """Collect summary statistics about the CS tree."""
    total_nodes = 0
    max_depth = 0
    leaf_count = 0
    queue: List[CognitiveStateNode] = [root]
    while queue:
        node = queue.pop()
        total_nodes += 1
        max_depth = max(max_depth, node.depth)
        if node.is_leaf:
            leaf_count += 1
        queue.extend(node.children)
    return {
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "leaf_count": leaf_count,
    }
