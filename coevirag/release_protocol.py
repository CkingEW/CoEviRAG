"""Release boundary checks; feature and navigation mathematics stay upstream."""

BENCHMARK_DATASETS = ("2wikimultihopqa", "hotpotqa", "musique", "mrqa_nq_fixed_1k")
DATASETS = (*BENCHMARK_DATASETS, "sample")


def validate_selector(checkpoint, final=False):
    from .query_plan_evidence import dynamic_query_plan_enabled
    dynamic_query_plan_enabled(checkpoint)
    if checkpoint.get("ablated_query_plan_features", False):
        raise ValueError("RQR feature ablations are not released")
    if final and (
        checkpoint.get("input_dim") != 53
        or not checkpoint.get("uses_selection_progress_features", False)
        or not checkpoint.get("uses_query_plan_features", False)
        or checkpoint.get("selector_mode") != "residual"
        or checkpoint.get("unary_mode") != "dpr"
    ):
        raise ValueError("Final retrieval requires the 53D fixed-RQR progress selector")


def validate_navigation_args(args):
    from .utils.misc_utils import string_to_bool
    if string_to_bool(args.repair_enabled) or string_to_bool(args.repair_apply_results):
        raise ValueError("Repair is excluded from CoEviRAG")
    if args.graph_variant != "unified_v1" or args.fact_traversal_mode != "role_aware":
        raise ValueError("Use the unified_v1 role-aware graph protocol")
    for name in ("fact_node_dropout_rate", "fact_edge_dropout_rate"):
        if getattr(args, name, 0):
            raise ValueError("Graph corruption experiments are excluded")
    for name in ("targeted_bridge_fact_dropout", "query_relation_fact_dropout"):
        if string_to_bool(getattr(args, name, "false")):
            raise ValueError("Graph corruption experiments are excluded")
