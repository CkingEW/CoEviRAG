# Modified for CoEviRAG: release scope, portable paths, or fixed-protocol guards.
import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:

    # --- 新增的RL配置 ---
    rl_learning_rate: float = 0.0001
    rl_gamma: float = 0.99
    rl_max_episode_steps: int = 10
    rl_hidden_dim: int = 256
    rl_reward_gold: float = 1.0
    rl_reward_step: float = -0.05
    rl_reward_mode: str = "legacy"
    rl_skip_completed_starts: bool = False
    rl_evidence_rank_loss_weight: float = 0.0
    rl_progress_scale: float = 2.0
    rl_terminal_progress_scale: float = 0.25
    rl_missing_gold_penalty: float = 0.25
    rl_complete_evidence_bonus: float = 0.0
    rl_advantage_mode: str = "episode_normalized"
    rl_baseline_momentum: float = 0.95
    rl_entropy_beta: float = 0.0
    rl_grad_clip: float = 0.0
    rl_update_batch_episodes: int = 1
    rl_seed_top_k: int = 12
    rl_train_seed_top_k: int = 3
    rl_seed_score_weight: float = 0.5
    rl_hierarchical_downstream_reward: bool = False
    rl_downstream_reward_weight: float = 1.0
    rl_downstream_complete_weight: float = 0.25
    rl_downstream_recall2_weight: float = 1.0
    rl_downstream_recall5_weight: float = 1.0
    rl_downstream_recall10_weight: float = 1.0
    rl_downstream_reward_clip: float = 1.0
    rl_downstream_seed_top_k: int = 6
    rl_downstream_credit_mode: str = "raw"
    rl_downstream_positive_bonus: float = 0.0
    rl_paragraph_seed_gate_enabled: bool = False
    rl_paragraph_seed_gate_threshold: float = 0.5
    rl_paragraph_seed_gate_min_keep: int = 0
    rl_seed_gate_cost: float = 0.01
    rl_seed_gate_loss_weight: float = 5.0
    train_paragraph_agent: bool = True
    use_paragraph_agent: bool = True
    paragraph_train_seed_top_k: int = 3
    paragraph_train_dpr_top_k: int = 10
    paragraph_train_fact_ratio: float = 0.5
    paragraph_train_fact_seed_source: str = "linking"
    paragraph_train_fact_search_seed_top_k: int = 4
    paragraph_train_fact_search_steps: int = 3

    
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="openai/gpt-oss-120b",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    llm_reasoning_effort: str = field(
        default="low",
        metadata={"help": "Reasoning effort for supported APIs; use 'none' to disable it."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."}
    )
    max_new_tokens: Union[None, int] = field(
        default=1024,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    execution_device: str = field(
        default="cuda",
        metadata={"help": "Torch device used by the fact and paragraph policy networks."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=1,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    llm_request_timeout: float = field(
        default=30.0,
        metadata={"help": "Timeout in seconds for each LLM API request."}
    )
    openie_batch_size: int = field(
        default=200,
        metadata={"help": "OpenIE passages processed between durable checkpoints."}
    )
    openie_num_workers: int = field(
        default=16,
        metadata={"help": "Concurrent workers used for online OpenIE requests."}
    )
    openie_pipeline_mode: str = field(
        default="staged",
        metadata={"help": "Use staged NER/triple batches or per-document pipelining."}
    )
    openie_max_new_passages: int = field(
        default=0,
        metadata={"help": "Maximum new passages processed in one indexing run; 0 is unlimited."}
    )
    openie_empty_retry_attempts: int = field(
        default=1,
        metadata={"help": "Corrective retries for empty or truncated OpenIE JSON."}
    )
    openie_batch_retry_attempts: int = field(
        default=3,
        metadata={"help": "Batch retries before an OpenIE API failure stops indexing."}
    )
    query_extraction_max_workers: int = field(
        default=4,
        metadata={"help": "Concurrent workers used for query OpenIE extraction."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    openie_results_path: str = ""
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="Qwen/Qwen3-Embedding-4B",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    graph_variant: str = field(
        default="",
        metadata={"help": "Optional graph artifact suffix used to isolate construction variants."}
    )
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    synonymy_edge_max_neighbors: int = field(
        default=100,
        metadata={"help": "Maximum accepted synonym neighbors per entity after thresholding."}
    )
    synonymy_edge_mutual_knn: bool = field(
        default=False,
        metadata={"help": "Keep an entity synonym edge only when both endpoints select each other."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    use_ppr_rerank: bool = field(
        default=False,
        metadata={"help": "Whether to include PPR scores in final reranking. Disabled by default for CoEviRAG."}
    )
    w_policy: float = field(
        default=0.2,
        metadata={"help": "Weight of graph/path policy score in final reranking."}
    )
    normalize_rerank_features: bool = field(
        default=False,
        metadata={"help": "Normalize heterogeneous reranking features within each query."}
    )
    normalize_path_by_length: bool = field(
        default=False,
        metadata={"help": "Use mean per-step path scores before query-level normalization."}
    )
    rerank_fusion_mode: str = field(
        default="score",
        metadata={"help": "Final candidate fusion: raw score, pure similarity, or query-level rank RRF."}
    )
    rank_fusion_rrf_k: int = field(
        default=60,
        metadata={"help": "RRF smoothing constant for rank-based candidate fusion."}
    )
    fact_head_protect_k: int = field(
        default=0,
        metadata={"help": "Reserve the first k final ranks for Fact-agent candidates."}
    )
    similarity_head_protect_k: int = field(
        default=0,
        metadata={"help": "Reserve the first k final ranks for dense-similarity candidates."}
    )
    query_conditioned_ranker_path: str = field(
        default="",
        metadata={"help": "Checkpoint for unified query-conditioned candidate reranking."}
    )
    query_conditioned_fact_budget: int = field(
        default=200,
        metadata={"help": "Maximum Fact-agent candidates passed to the unified ranker."}
    )
    query_conditioned_paragraph_budget: int = field(
        default=200,
        metadata={"help": "Maximum Paragraph-agent candidates passed to the unified ranker."}
    )
    query_conditioned_repair_budget: int = field(
        default=50,
        metadata={"help": "Maximum verified Repair candidates passed to the unified ranker."}
    )
    query_conditioned_similarity_budget: int = field(
        default=0,
        metadata={"help": "Maximum global dense-similarity candidates added as a safety channel."}
    )
    set_aware_selector_path: str = field(
        default="",
        metadata={"help": "Optional semantic anchor-bridge selector applied after unified ranking."}
    )
    set_aware_selection_mode: str = field(
        default="select2",
        metadata={"help": "Evidence selection mode: select2, dynamic_setmax, or dynamic_setmax_recovery."}
    )
    set_aware_anchor_mode: str = field(
        default="unary",
        metadata={"help": "Anchor/selector prior: unary ranker score or DPR similarity."}
    )
    retrieval_source_mode: str = field(
        default="dual",
        metadata={"help": "Candidate source isolation: dual, fact_only, or paragraph_only."}
    )
    source_diagnostics: bool = field(
        default=False,
        metadata={"help": "Report source-level retrieval and complete-evidence diagnostics."}
    )
    query_fact_score_mode: str = field(
        default="concat",
        metadata={"help": "Structured query fact scoring: concat, slot_max, slot_mean, or slot_hybrid."}
    )
    query_slot_blend_alpha: float = field(
        default=0.5,
        metadata={"help": "Slot-max weight in slot_hybrid structured query scoring."}
    )
    w_query: float = field(
        default=0.5,
        metadata={"help": "Raw-query weight versus structured-query fact scoring."}
    )
    cache_static_retrieval: bool = field(
        default=True,
        metadata={"help": "Cache query-to-fact and query-to-passage scores and rankings across epochs."}
    )
    optimize_retrieval: bool = field(
        default=True,
        metadata={"help": "Cache graph neighbors and batch GPU-to-CPU score transfers."}
    )
    optimize_static_graph_ops: bool = field(
        default=True,
        metadata={"help": "Cache immutable graph computations and bulk-transfer search scores without changing rankings."}
    )
    precompute_paragraph_knn: bool = field(
        default=False,
        metadata={"help": "Precompute paragraph semantic Top-K neighbors in GPU batches."}
    )
    paragraph_knn_batch_size: int = field(
        default=256,
        metadata={"help": "Query batch size for GPU paragraph KNN precomputation."}
    )
    batch_fact_beam_scoring: bool = field(
        default=True,
        metadata={"help": "Score all active Fact beams in one policy-network forward per step."}
    )
    reuse_fact_path_prefix: bool = field(
        default=True,
        metadata={"help": "Reuse already mapped Fact path prefixes without changing candidates or scores."}
    )
    batch_fact_path_mapping: bool = field(
        default=False,
        metadata={"help": "Map sibling Fact paths together while preserving update order and scores."}
    )
    fact_traversal_mode: Literal["undirected", "hard_directed", "role_aware"] = field(
        default="undirected",
        metadata={"help": "Fact-KG traversal direction used consistently by training and inference."}
    )
    w_fact_policy: float = field(
        default=0.15,
        metadata={"help": "Normalized Fact-agent path-score weight."}
    )
    w_paragraph_policy: float = field(
        default=0.15,
        metadata={"help": "Normalized Paragraph-agent path-score weight."}
    )
    w_sim: float = field(
        default=0.6,
        metadata={"help": "Weight of query-passage similarity in final reranking."}
    )
    w_ppr: float = field(
        default=0.0,
        metadata={"help": "Weight of PPR score in final reranking."}
    )
    w_repair_confidence: float = field(
        default=0.1,
        metadata={"help": "Weight of verified repair confidence in final reranking."}
    )
    w_source_agreement: float = field(
        default=0.05,
        metadata={"help": "Bonus for passages found by multiple retrieval agents."}
    )
    use_learned_coordinator: bool = field(
        default=False,
        metadata={"help": "Whether to use a trained lightweight coordinator gate for final reranking."}
    )
    coordinator_blend_alpha: float = field(
        default=0.1,
        metadata={"help": "Residual blend weight for learned coordinator logits on top of heuristic scores."}
    )
    coordinator_protect_top_k: int = field(
        default=0,
        metadata={"help": "Keep top-k heuristic candidates unchanged before applying learned coordinator residual."}
    )
    coordinator_margin_gate_top_k: int = field(
        default=0,
        metadata={"help": "Disable coordinator per query when the heuristic top-k boundary is not residual-safe."}
    )
    coordinator_margin_gate_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum safe top-k boundary margin; <=0 uses coordinator_blend_alpha."}
    )
    
    
    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )

    # CoEviRAG retrieval attributes
    paragraph_graph_topk: int = field(
        default=3,
        metadata={"help": "KNN degree for the implicit paragraph-only graph."}
    )
    fact_return_paragraph_expansion: bool = field(
        default=True,
        metadata={"help": "Expand mapped Fact passages through semantic KNN before Paragraph-agent seeding."}
    )
    use_evidence_paragraph_graph: bool = field(
        default=False,
        metadata={"help": "Whether paragraph neighbors include shared-entity and shared-fact evidence edges."}
    )
    evidence_paragraph_max_candidates: int = field(
        default=80,
        metadata={"help": "Max evidence-aware paragraph candidates considered per paragraph node."}
    )
    evidence_paragraph_max_entity_support: int = field(
        default=80,
        metadata={"help": "Skip entity evidence edges when an entity appears in more than this many passages."}
    )
    paragraph_agent_seed_top_k: int = field(
        default=20,
        metadata={"help": "Number of DPR seed paragraphs for the paragraph agent."}
    )
    paragraph_seed_mode: str = field(
        default="hybrid",
        metadata={"help": "Paragraph seeds: hybrid, dpr_only, or fact_only."}
    )
    paragraph_fact_seed_score_mode: str = field(
        default="fixed",
        metadata={"help": "Score Fact-mapped paragraph seeds with fixed 0.5 or normalized Fact confidence."}
    )
    paragraph_agent_max_steps: int = field(
        default=2,
        metadata={"help": "Maximum expansion steps for the paragraph agent."}
    )
    paragraph_agent_beam_width: int = field(
        default=20,
        metadata={"help": "Beam width for paragraph-agent graph search."}
    )
    fact_agent_weight: float = field(
        default=0.55,
        metadata={"help": "Final score weight for fact-agent evidence."}
    )
    paragraph_agent_weight: float = field(
        default=0.35,
        metadata={"help": "Final score weight for paragraph-agent evidence."}
    )
    repair_enabled: bool = field(
        default=False,
        metadata={"help": "Compatibility field; must remain false."}
    )
    repair_apply_results: bool = field(
        default=False,
        metadata={"help": "Compatibility field; must remain false."}
    )
    fact_edge_dropout_rate: float = field(
        default=0.0,
        metadata={"help": "In-memory deterministic dropout rate for entity-fact edges used in robustness experiments."}
    )
    fact_edge_dropout_seed: int = field(
        default=42,
        metadata={"help": "Seed used to select deterministic entity-fact edge dropout."}
    )
    fact_node_dropout_rate: float = field(
        default=0.0,
        metadata={"help": "In-memory deterministic dropout rate for complete OpenIE fact nodes."}
    )
    fact_node_dropout_seed: int = field(
        default=42,
        metadata={"help": "Seed used to select deterministic complete-fact dropout."}
    )
    targeted_bridge_fact_dropout: bool = field(
        default=False,
        metadata={"help": "Remove gold-document bridge facts in memory to simulate missing OpenIE triples."}
    )
    targeted_bridge_passage_mask: bool = field(
        default=False,
        metadata={"help": "Hide bridge-supporting gold passages from base agents while keeping them available to Repair."}
    )
    targeted_bridge_facts_per_query: int = field(
        default=1,
        metadata={"help": "Maximum number of bridge fact nodes removed per evaluation query."}
    )
    query_relation_fact_dropout: bool = field(
        default=False,
        metadata={"help": "Remove query-entity/relation-matched facts without using gold annotations."}
    )
    query_relation_passage_mask: bool = field(
        default=False,
        metadata={"help": "Hide source passages of query-relation-matched facts from base retrieval."}
    )
    query_relation_facts_per_query: int = field(
        default=1,
        metadata={"help": "Maximum number of query-relation-matched facts removed per evaluation query."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.repair_enabled or self.repair_apply_results:
            raise ValueError("Repair is excluded from CoEviRAG")
        if self.openie_mode != "online":
            raise ValueError("Only remote OpenAI-compatible OpenIE is included")
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
