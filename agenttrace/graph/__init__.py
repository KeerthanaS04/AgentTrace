"""Graph builder and queries over span data-flow DAGs."""

from agenttrace.graph.alignment import AlignedSpan, Alignment, AlignmentError, align_runs
from agenttrace.graph.builder import (
    GraphError,
    build_graph,
    data_subgraph,
    edges_of_kind,
    executed_topology,
    node_topology,
    output_hash,
    span_status,
)
from agenttrace.graph.queries import (
    CriticalPath,
    Divergence,
    ancestors,
    critical_path,
    descendants,
    downstream,
    final_spans,
    first_divergence,
    paths_to_final,
    topological_order,
)

__all__ = [
    "AlignedSpan",
    "Alignment",
    "AlignmentError",
    "align_runs",
    "CriticalPath",
    "Divergence",
    "GraphError",
    "ancestors",
    "build_graph",
    "critical_path",
    "data_subgraph",
    "descendants",
    "downstream",
    "edges_of_kind",
    "executed_topology",
    "final_spans",
    "first_divergence",
    "node_topology",
    "output_hash",
    "paths_to_final",
    "span_status",
    "topological_order",
]
