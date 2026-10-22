# Modified for CoEviRAG: release scope, portable paths, or fixed-protocol guards.
import json
import os
import time
import logging
import ast
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import re
import hashlib
import heapq
import math
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Assuming these imports exist in the user's environment structure
from coevirag.llm import _get_llm_class, BaseLLM
from coevirag.embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from coevirag.embedding_store import EmbeddingStore
from coevirag.information_extraction import OpenIE
from coevirag.evaluation.retrieval_eval import RetrievalRecall
from coevirag.evaluation.qa_eval import QAExactMatch, QAF1Score
from coevirag.prompts.linking import get_query_instruction
from coevirag.prompts.prompt_template_manager import PromptTemplateManager
from coevirag.prompts.profiles import qa_prompt_for_model
from coevirag.utils.qa_utils import parse_qa_answer
from coevirag.utils.misc_utils import *
from coevirag.utils.misc_utils import NerRawOutput, TripleRawOutput
from coevirag.utils.embed_utils import retrieve_knn
from coevirag.utils.typing import Triple
from coevirag.utils.config_utils import BaseConfig
from coevirag.training_evidence import local_evidence_utilities, listwise_evidence_loss
from coevirag.query_plan_evidence import (
    append_selection_progress,
    dynamic_query_plan_enabled,
    query_plan_slot_scores,
    query_plan_state_features,
    valid_openie_triples,
)

logger = logging.getLogger(__name__)

class PolicyNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        """
        Initializes the policy network with an LSTM for history and MLPs for state and action scoring.
        """
        super(PolicyNetwork, self).__init__()
        self.history_rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.state_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.seed_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Preserve pre-gate checkpoint behavior until the gate is trained.
        nn.init.zeros_(self.seed_gate[-1].weight)
        nn.init.constant_(self.seed_gate[-1].bias, 1.0)
    def forward(self, current_node_emb, query_emb, history_embs, neighbor_embs):
        """
        Forward pass to compute action probabilities for neighbor nodes based on the current state.
        """
        _, (h_n, _) = self.history_rnn(history_embs)
        history_context = h_n.squeeze(0)
        state_input = torch.cat([current_node_emb, query_emb, history_context], dim=1)
        state_context = self.state_mlp(state_input)
        if neighbor_embs.shape[0] == 0:
            return Categorical(torch.tensor([]))
        state_context_repeated = state_context.repeat(neighbor_embs.shape[0], 1)
        action_input = torch.cat([state_context_repeated, neighbor_embs], dim=1)
        scores = self.action_scorer(action_input).squeeze(1)
        action_probs = F.softmax(scores, dim=-1)
        return Categorical(action_probs)

    def seed_keep_logit(self, seed_emb, query_emb):
        """Score the high-level KEEP action for one Fact navigation seed."""
        gate_input = torch.cat([seed_emb, query_emb], dim=1)
        return self.seed_gate(gate_input).squeeze(1)

    def action_log_probs_batch(
            self,
            current_node_embs: torch.Tensor,
            query_embs: torch.Tensor,
            history_embs: torch.Tensor,
            neighbor_emb_groups: List[torch.Tensor]) -> List[torch.Tensor]:
        """Score variable-sized action sets for several beams in one forward pass."""
        _, (h_n, _) = self.history_rnn(history_embs)
        history_context = h_n.squeeze(0)
        state_input = torch.cat(
            [current_node_embs, query_embs, history_context],
            dim=1,
        )
        state_context = self.state_mlp(state_input)
        neighbor_counts = [group.shape[0] for group in neighbor_emb_groups]
        repeated_states = torch.repeat_interleave(
            state_context,
            torch.tensor(neighbor_counts, device=state_context.device),
            dim=0,
        )
        all_neighbors = torch.cat(neighbor_emb_groups, dim=0)
        action_input = torch.cat([repeated_states, all_neighbors], dim=1)
        raw_scores = self.action_scorer(action_input).squeeze(1)
        return [
            F.log_softmax(group_scores, dim=0)
            for group_scores in torch.split(raw_scores, neighbor_counts)
        ]


class RLAgent:
    def __init__(
            self,
            policy_network,
            learning_rate: float,
            gamma: float,
            entropy_beta: float = 0.0,
            grad_clip: float = 0.0,
            advantage_mode: str = "episode_normalized",
            baseline_momentum: float = 0.95):
        """
        Initializes the RL Agent with a policy network, optimizer, and reward discount factor.
        """
        self.policy = policy_network
        self.device = next(self.policy.parameters()).device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.grad_clip = grad_clip
        self.advantage_mode = advantage_mode
        self.baseline_momentum = baseline_momentum
        self.return_baseline = 0.0
        self.baseline_initialized = False
        self.saved_log_probs = []
        self.saved_entropies = []
        self.rewards = []
        self.saved_evidence_rank_losses = []
        self.last_action_logits = None
        self.last_evidence_rank_states = 0
    
    def select_action(self, state_embs):
        """
        Select an action and retain its differentiable statistics for REINFORCE.
        """
        self.last_action_logits = None
        current_node_emb, query_emb, history_embs, neighbor_embs = state_embs
        if neighbor_embs.shape[0] == 0: return None, None
        
        dist = self.policy(current_node_emb, query_emb, history_embs, neighbor_embs)
        if dist.probs.nelement() == 0: return None, None
        
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)
        
        self.saved_log_probs.append(log_prob) 
        self.saved_entropies.append(dist.entropy())
        self.last_action_logits = dist.logits
        # Only the sampled index is needed by Python. Converting log_prob to a
        # scalar here caused an additional CUDA synchronization every step.
        return action_index.item(), None

    def _record_evidence_rank_loss(self, rag, query, neighbors, gold_docs, paragraph=False):
        weight = float(getattr(rag.global_config, "rl_evidence_rank_loss_weight", 0.0))
        if weight == 0.0:
            return
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Evidence rank loss weight must be finite and nonnegative")
        if self.last_action_logits is None:
            return
        utilities = local_evidence_utilities(rag, query, neighbors, gold_docs, paragraph)
        loss = listwise_evidence_loss(self.last_action_logits, utilities)
        if loss is not None:
            self.saved_evidence_rank_losses.append(weight * loss)
            self.last_evidence_rank_states += 1

    def seed_gate_loss(
            self,
            seed_emb: torch.Tensor,
            query_emb: torch.Tensor,
            keep_advantage: float,
            loss_weight: float = 1.0):
        """Train KEEP/SKIP from the measured counterfactual seed utility."""
        logit = self.policy.seed_keep_logit(seed_emb, query_emb)
        target = torch.ones_like(logit) if keep_advantage > 0.0 else torch.zeros_like(logit)
        sample_weight = min(1.0, abs(float(keep_advantage)))
        if sample_weight <= 0.0 or loss_weight <= 0.0:
            return None
        return (
            F.binary_cross_entropy_with_logits(logit, target)
            * sample_weight
            * float(loss_weight)
        )

    #Reward Shaping
    def run_episode(self, coevirag_instance, start_node, query, gold_docs, gold_doc_embs, 
                        max_steps, reward_gold, reward_step):
        """
        Runs a target-driven episode where rewards are based on similarity to gold documents (the answer).
        """
        self.last_episode_start_skipped = False
        self.last_evidence_rank_states = 0
        path = [start_node]
        current_node = start_node
        
        # 1. Prepare Query Embedding 
        query_emb_np = coevirag_instance.query_to_embedding['passage'].get(query, None)
        if query_emb_np is None:
             query_emb_np = coevirag_instance.embedding_model.batch_encode(query, 
                                                             instruction=get_query_instruction('query_to_passage'), 
                                                             norm=True)
             coevirag_instance.query_to_embedding['passage'][query] = query_emb_np
        query_emb = torch.from_numpy(query_emb_np).float().view(1, -1).to(self.device)

        # 2. Prepare Start Node Embedding                 
        current_node_emb = coevirag_instance.get_embedding_for_node(current_node)
        if current_node_emb is not None:
            current_node_emb = current_node_emb.view(1, -1)
        else:
            logger.warning(f"Start node {start_node} has NO embedding! Aborting episode.")
            return path
           
        # Calculate current distance to target (Baseline)
        if gold_doc_embs is not None and len(gold_doc_embs) > 0:
            gold_doc_embs = gold_doc_embs.to(self.device)
            max_sim_to_gold_current = torch.max(F.cosine_similarity(current_node_emb, gold_doc_embs)).item()
        else:
            max_sim_to_gold_current = 0.0

        reward_mode = getattr(coevirag_instance.global_config, "rl_reward_mode", "legacy")
        gold_doc_set = set(gold_docs or [])
        covered_gold_docs = set()

        def gold_docs_for_node(node_key):
            matched = set()
            for passage_key in coevirag_instance.get_supporting_passages_for_node(node_key):
                try:
                    passage_content = coevirag_instance.chunk_embedding_store.get_row(passage_key)["content"]
                except Exception:
                    continue
                if passage_content in gold_doc_set:
                    matched.add(passage_content)
            return matched

        if reward_mode == "coverage_progress":
            covered_gold_docs.update(gold_docs_for_node(start_node))
            # A solved start supplies no action credit; retain legacy/hierarchical behavior.
            if (getattr(coevirag_instance.global_config, "rl_skip_completed_starts", False)
                    and not getattr(coevirag_instance.global_config, "rl_hierarchical_downstream_reward", False)
                    and gold_doc_set and covered_gold_docs >= gold_doc_set):
                self.last_episode_start_skipped = True
                return path

        found_gold = False
        initial_max_sim = max_sim_to_gold_current
        max_sim_reached = max_sim_to_gold_current 

        for step in range(max_steps):
            # --- Action Selection ---
            try:
                neighbor_names = [
                    name for name in coevirag_instance._get_fact_search_neighbors(
                        current_node,
                        query,
                        previous_node=path[-2] if len(path) > 1 else None,
                    )
                    if name not in path
                ]
                if not neighbor_names: break
                
                neighbor_embs_list, valid_neighbor_names = [], []
                for n in neighbor_names:
                    emb = coevirag_instance.get_embedding_for_node(n)
                    if emb is not None:
                        neighbor_embs_list.append(emb)
                        valid_neighbor_names.append(n)
                
                if not valid_neighbor_names: break
                neighbor_names = valid_neighbor_names
                
                neighbor_embs = torch.stack(neighbor_embs_list) 
                history_embs_list = [coevirag_instance.get_embedding_for_node(n) for n in path]
                valid_history_embs = [e for e in history_embs_list if e is not None]
                if not valid_history_embs: continue
                history_embs = torch.stack(valid_history_embs).unsqueeze(0)
                
                state_embs = (current_node_emb, query_emb, history_embs, neighbor_embs)
                action_info = self.select_action(state_embs)
                if action_info is None: break
                action_index, _ = action_info
                
                next_node = neighbor_names[action_index]
                next_node_emb = neighbor_embs_list[action_index].view(1, -1)
            
            except Exception as e:
                logger.debug(f"run_episode Error: {e}"); break

            if reward_mode == "coverage_progress":
                self._record_evidence_rank_loss(coevirag_instance, query, neighbor_names,
                                                gold_doc_set - covered_gold_docs)

            # --- Reward Calculation (Target-Driven) ---
            
            # 1. Calculate similarity of new node to target
            if gold_doc_embs is not None and len(gold_doc_embs) > 0:
                max_sim_to_gold_next = torch.max(F.cosine_similarity(next_node_emb, gold_doc_embs)).item()
            else:
                max_sim_to_gold_next = 0.0
            
            # 2. Progress reward
            progress = max_sim_to_gold_next - max_sim_to_gold_current
            if reward_mode == "coverage_progress":
                progress_scale = getattr(coevirag_instance.global_config, "rl_progress_scale", 2.0)
                reward = reward_step + progress_scale * progress
                newly_covered = gold_docs_for_node(next_node) - covered_gold_docs
                if newly_covered:
                    early_discount = 1.0 / math.log2(step + 2.0)
                    reward += reward_gold * early_discount * len(newly_covered)
                    covered_gold_docs.update(newly_covered)
                found_gold = bool(gold_doc_set) and covered_gold_docs >= gold_doc_set
            else:
                progress_reward = max(0, progress)
                reward = reward_step + progress_reward * 2.0
                supporting_passages = coevirag_instance.get_supporting_passages_for_node(next_node)
                for passage_key in supporting_passages:
                    passage_content = coevirag_instance.chunk_embedding_store.get_row(passage_key)["content"]
                    if passage_content in gold_docs:
                        reward += reward_gold
                        found_gold = True
                        break

            self.rewards.append(reward)
            path.append(next_node)
            if found_gold:
                break

            # --- State Transition ---
            current_node = next_node
            current_node_emb = next_node_emb
            max_sim_to_gold_current = max_sim_to_gold_next
            max_sim_reached = max(max_sim_reached, max_sim_to_gold_next)

        # --- Soft Terminal Reward ---
        if reward_mode == "coverage_progress" and self.rewards:
            progress_scale = getattr(
                coevirag_instance.global_config,
                "rl_terminal_progress_scale",
                0.25,
            )
            missing_penalty = getattr(
                coevirag_instance.global_config,
                "rl_missing_gold_penalty",
                0.25,
            )
            terminal_progress = max(0.0, max_sim_reached - initial_max_sim)
            self.rewards[-1] += reward_gold * progress_scale * terminal_progress
            if gold_doc_set:
                missing_ratio = 1.0 - len(covered_gold_docs) / len(gold_doc_set)
                self.rewards[-1] -= reward_gold * missing_penalty * missing_ratio
                if covered_gold_docs >= gold_doc_set:
                    complete_bonus = getattr(
                        coevirag_instance.global_config,
                        "rl_complete_evidence_bonus",
                        0.0,
                    )
                    self.rewards[-1] += reward_gold * complete_bonus
        elif not found_gold and gold_doc_embs is not None and self.rewards:
            if max_sim_reached > 0: 
                soft_reward = reward_gold * max_sim_reached
                self.rewards[-1] += soft_reward

        return path

    def run_episode_query(self, coevirag_instance, start_node, query, gold_docs, gold_doc_embs, 
                            max_steps, reward_gold, reward_step):
            """
            Runs an episode where rewards are driven by query similarity (Query-Driven Progress). 
            Used primarily for ablation studies.
            """
            path = [start_node]
            current_node = start_node
            
            query_emb_np = coevirag_instance.query_to_embedding['passage'].get(query, None)
            if query_emb_np is None:
                 query_emb_np = coevirag_instance.embedding_model.batch_encode(query, 
                                                                 instruction=get_query_instruction('query_to_passage'), 
                                                                 norm=True)
                 coevirag_instance.query_to_embedding['passage'][query] = query_emb_np
            
            if isinstance(query_emb_np, np.ndarray):
                query_emb = torch.from_numpy(query_emb_np).float().view(1, -1).to(self.device)
            else:
                query_emb = query_emb_np.view(1, -1).to(self.device)
    
            current_node_emb = coevirag_instance.get_embedding_for_node(current_node) 
            if current_node_emb is not None:
                current_node_emb = current_node_emb.view(1, -1) 
            else:
                logger.warning(f"Start node {start_node} has NO embedding! Aborting episode.")
                return path
            
            sim_current = F.cosine_similarity(current_node_emb, query_emb, dim=1).item()
    
            for step in range(max_steps):
                try:
                    neighbor_names = [
                        name for name in coevirag_instance._get_fact_search_neighbors(
                            current_node,
                            query,
                            previous_node=path[-2] if len(path) > 1 else None,
                        )
                        if name not in path
                    ]
                    if not neighbor_names: break
                    
                    neighbor_embs_list = [] 
                    valid_neighbor_names = []
                    for n in neighbor_names:
                        emb = coevirag_instance.get_embedding_for_node(n)
                        if emb is not None:
                            neighbor_embs_list.append(emb)
                            valid_neighbor_names.append(n)
                    
                    if not valid_neighbor_names: break
                    neighbor_names = valid_neighbor_names
                    
                    neighbor_embs = torch.stack(neighbor_embs_list).to(self.device) 
                    
                    history_embs_list = [coevirag_instance.get_embedding_for_node(n) for n in path]
                    valid_history_embs = [e for e in history_embs_list if e is not None]
                    if not valid_history_embs: continue
                    history_embs = torch.stack(valid_history_embs).unsqueeze(0).to(self.device)
                    
                    state_embs = (current_node_emb, query_emb, history_embs, neighbor_embs)
                    action_info = self.select_action(state_embs)
                    if action_info is None: break
                    action_index, _ = action_info
                    
                    next_node = neighbor_names[action_index]
                    next_node_emb = neighbor_embs_list[action_index].view(1, -1) 
                
                except Exception as e:
                    logger.debug(f"run_episode Error: {e}"); break
    
                # --- Reward Calculation ---
                sim_next = F.cosine_similarity(next_node_emb, query_emb, dim=1).item()
                progress_reward = max(0, sim_next - sim_current)
                
                reward = reward_step 
                reward += progress_reward * 1.0

                supporting_passages = coevirag_instance.get_supporting_passages_for_node(next_node)
                found_gold = False
                for passage_key in supporting_passages:
                    passage_content = coevirag_instance.chunk_embedding_store.get_row(passage_key)["content"]
                    if passage_content in gold_docs:
                        reward += reward_gold
                        self.rewards.append(reward)
                        path.append(next_node)
                        found_gold = True
                        break
                if found_gold:
                    break
                        
                self.rewards.append(reward)
    
                # --- State Transition ---
                path.append(next_node)
                current_node = next_node
                current_node_emb = next_node_emb 
                sim_current = sim_next 

            return path     

    def run_paragraph_episode(self, coevirag_instance, start_node, query, gold_docs, gold_doc_embs,
                              max_steps, reward_gold, reward_step):
        """Train on the implicit paragraph-only KNN graph."""
        self.last_episode_start_skipped = False
        self.last_evidence_rank_states = 0
        path = [start_node]
        current_node = start_node
        gold_doc_set = set(gold_docs or [])

        query_emb = coevirag_instance._get_query_embedding_tensor(query).to(self.device)
        current_node_emb = coevirag_instance.get_embedding_for_node(current_node)
        if current_node_emb is None:
            logger.warning(f"Paragraph start node {start_node} has NO embedding! Aborting episode.")
            return path
        current_node_emb = current_node_emb.view(1, -1)

        if gold_doc_embs is not None and len(gold_doc_embs) > 0:
            gold_doc_embs = gold_doc_embs.to(self.device)
            max_sim_to_gold_current = torch.max(F.cosine_similarity(current_node_emb, gold_doc_embs)).item()
        else:
            max_sim_to_gold_current = F.cosine_similarity(current_node_emb, query_emb, dim=1).item()

        reward_mode = getattr(coevirag_instance.global_config, "rl_reward_mode", "legacy")
        covered_gold_docs = set()
        if reward_mode == "coverage_progress":
            try:
                start_content = coevirag_instance.chunk_embedding_store.get_row(start_node)["content"]
                if start_content in gold_doc_set:
                    covered_gold_docs.add(start_content)
            except Exception:
                pass
            if (getattr(coevirag_instance.global_config, "rl_skip_completed_starts", False)
                    and not getattr(coevirag_instance.global_config, "rl_hierarchical_downstream_reward", False)
                    and gold_doc_set and covered_gold_docs >= gold_doc_set):
                self.last_episode_start_skipped = True
                return path

        found_gold = False
        initial_max_sim = max_sim_to_gold_current
        max_sim_reached = max_sim_to_gold_current
        neighbor_top_k = getattr(coevirag_instance.global_config, "paragraph_graph_topk", 3)

        for step in range(max_steps):
            try:
                neighbor_names = [
                    key for key, _ in coevirag_instance._get_paragraph_neighbors(current_node, top_k=neighbor_top_k)
                    if key not in path
                ]
                if not neighbor_names:
                    break

                neighbor_embs_list, valid_neighbor_names = [], []
                for n in neighbor_names:
                    emb = coevirag_instance.get_embedding_for_node(n)
                    if emb is not None:
                        neighbor_embs_list.append(emb)
                        valid_neighbor_names.append(n)
                if not valid_neighbor_names:
                    break

                neighbor_embs = torch.stack(neighbor_embs_list)
                history_embs_list = [coevirag_instance.get_embedding_for_node(n) for n in path]
                valid_history_embs = [e for e in history_embs_list if e is not None]
                if not valid_history_embs:
                    break
                history_embs = torch.stack(valid_history_embs).unsqueeze(0)

                action_info = self.select_action((current_node_emb, query_emb, history_embs, neighbor_embs))
                if action_info is None:
                    break
                action_index, _ = action_info

                next_node = valid_neighbor_names[action_index]
                next_node_emb = neighbor_embs_list[action_index].view(1, -1)
            except Exception as e:
                logger.debug(f"run_paragraph_episode Error: {e}")
                break

            if reward_mode == "coverage_progress":
                self._record_evidence_rank_loss(coevirag_instance, query, valid_neighbor_names,
                                                gold_doc_set - covered_gold_docs, paragraph=True)

            if gold_doc_embs is not None and len(gold_doc_embs) > 0:
                max_sim_to_gold_next = torch.max(F.cosine_similarity(next_node_emb, gold_doc_embs)).item()
            else:
                max_sim_to_gold_next = F.cosine_similarity(next_node_emb, query_emb, dim=1).item()

            progress = max_sim_to_gold_next - max_sim_to_gold_current
            if reward_mode == "coverage_progress":
                progress_scale = getattr(coevirag_instance.global_config, "rl_progress_scale", 2.0)
                reward = reward_step + progress_scale * progress
            else:
                reward = reward_step + max(0, progress) * 2.0

            try:
                passage_content = coevirag_instance.chunk_embedding_store.get_row(next_node)["content"]
                if reward_mode == "coverage_progress" and passage_content in gold_doc_set:
                    newly_covered = passage_content not in covered_gold_docs
                    if newly_covered:
                        early_discount = 1.0 / math.log2(step + 2.0)
                        reward += reward_gold * early_discount
                        covered_gold_docs.add(passage_content)
                    found_gold = bool(gold_doc_set) and covered_gold_docs >= gold_doc_set
                elif passage_content in gold_doc_set:
                    reward += reward_gold
                    found_gold = True
            except Exception:
                pass

            self.rewards.append(reward)
            path.append(next_node)
            if found_gold:
                break

            current_node = next_node
            current_node_emb = next_node_emb
            max_sim_to_gold_current = max_sim_to_gold_next
            max_sim_reached = max(max_sim_reached, max_sim_to_gold_next)

        if reward_mode == "coverage_progress" and self.rewards:
            progress_scale = getattr(
                coevirag_instance.global_config,
                "rl_terminal_progress_scale",
                0.25,
            )
            missing_penalty = getattr(
                coevirag_instance.global_config,
                "rl_missing_gold_penalty",
                0.25,
            )
            terminal_progress = max(0.0, max_sim_reached - initial_max_sim)
            self.rewards[-1] += reward_gold * progress_scale * terminal_progress
            if gold_doc_set:
                missing_ratio = 1.0 - len(covered_gold_docs) / len(gold_doc_set)
                self.rewards[-1] -= reward_gold * missing_penalty * missing_ratio
                if covered_gold_docs >= gold_doc_set:
                    complete_bonus = getattr(
                        coevirag_instance.global_config,
                        "rl_complete_evidence_bonus",
                        0.0,
                    )
                    self.rewards[-1] += reward_gold * complete_bonus
        elif not found_gold and gold_doc_embs is not None and self.rewards:
            self.rewards[-1] += reward_gold * max(0.0, max_sim_reached)

        return path

    def consume_episode_loss(self):
        """Build one REINFORCE loss and clear the episode buffers."""
        if not self.saved_log_probs:
            self.rewards = []
            self.saved_entropies = []
            self.saved_evidence_rank_losses = []
            self.last_action_logits = None
            return None

        R = 0
        policy_loss_terms = []
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if self.advantage_mode == "episode_normalized":
            if len(returns) > 1:
                advantages = (returns - returns.mean()) / (returns.std() + 1e-9)
            else:
                advantages = returns - returns.mean()
        elif self.advantage_mode == "ema_baseline":
            baseline = self.return_baseline if self.baseline_initialized else 0.0
            advantages = returns - baseline
            episode_mean = float(returns.mean().detach().cpu())
            if self.baseline_initialized:
                self.return_baseline = (
                    self.baseline_momentum * self.return_baseline
                    + (1.0 - self.baseline_momentum) * episode_mean
                )
            else:
                self.return_baseline = episode_mean
                self.baseline_initialized = True
        elif self.advantage_mode == "raw":
            advantages = returns
        else:
            raise ValueError(f"Unsupported RL advantage mode: {self.advantage_mode}")

        for idx, (log_prob, advantage) in enumerate(zip(self.saved_log_probs, advantages)):
            entropy_bonus = self.saved_entropies[idx] if idx < len(self.saved_entropies) else 0.0
            policy_loss_terms.append(-log_prob * advantage - self.entropy_beta * entropy_bonus)

        auxiliary_loss = (
            torch.stack(self.saved_evidence_rank_losses).mean()
            if self.saved_evidence_rank_losses else None
        )
        self.saved_evidence_rank_losses = []
        self.last_action_logits = None
        self.rewards = []
        self.saved_log_probs = []
        self.saved_entropies = []
        if not policy_loss_terms:
            return None
        loss = torch.stack(policy_loss_terms).sum()
        return loss if auxiliary_loss is None else loss + auxiliary_loss

    def take_episode_trace(self) -> Dict[str, Any]:
        """Detach one episode from the shared buffers while retaining its graph."""
        trace = {
            "rewards": list(self.rewards),
            "saved_log_probs": list(self.saved_log_probs),
            "saved_entropies": list(self.saved_entropies),
            "evidence_rank_losses": list(self.saved_evidence_rank_losses),
        }
        self.rewards = []
        self.saved_log_probs = []
        self.saved_entropies = []
        self.saved_evidence_rank_losses = []
        self.last_action_logits = None
        return trace

    def consume_episode_trace(
            self,
            trace: Dict[str, Any],
            terminal_bonus: float = 0.0):
        """Build an episode loss after adding delayed hierarchical credit."""
        self.rewards = list(trace.get("rewards", []))
        self.saved_log_probs = list(trace.get("saved_log_probs", []))
        self.saved_entropies = list(trace.get("saved_entropies", []))
        self.saved_evidence_rank_losses = list(trace.get("evidence_rank_losses", []))
        if self.rewards:
            self.rewards[-1] += float(terminal_bonus)
        return self.consume_episode_loss()

    def apply_policy_losses(self, policy_losses):
        """Apply one optimizer step for a mini-batch of complete episodes."""
        valid_losses = [loss for loss in policy_losses if loss is not None]
        if not valid_losses:
            return
        self.optimizer.zero_grad()
        policy_loss = torch.stack(valid_losses).mean()
        policy_loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

    def update_policy(self):
        """Compatibility path for one optimizer step per episode."""
        policy_loss = self.consume_episode_loss()
        if policy_loss is not None:
            self.apply_policy_losses([policy_loss])

# ==============================================================================
#  CoEviRAG Main Class
# ==============================================================================

class CoEviRAG:
    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None,
                 *args, **kwargs):
        """
        Initializes the CoEviRAG instance with reinforcement learning parameters.
        Sets up the device, working directory, and query triplet cache.
        """
        # --- Base Initialization Logic ---
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        if getattr(self.global_config, "repair_enabled", False) or getattr(self.global_config, "repair_apply_results", False):
            raise ValueError("Repair is excluded from CoEviRAG")

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"CoEviRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)
        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = (
            self.global_config.openie_results_path
            or os.path.join(
                self.global_config.save_dir,
                f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json',
            )
        )

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0

        self.ent_node_to_chunk_ids = None

        # --- RL Specific Initialization ---
        self.rl_learning_rate = getattr(self.global_config, 'rl_learning_rate', 1e-4)
        self.rl_gamma = getattr(self.global_config, 'rl_gamma', 0.99)
        self.rl_max_episode_steps = getattr(self.global_config, 'rl_max_episode_steps', 15)
        self.rl_hidden_dim = getattr(self.global_config, 'rl_hidden_dim', 256)
        self.rl_reward_gold = getattr(self.global_config, 'rl_reward_gold', 1.0)
        self.rl_reward_step = getattr(self.global_config, 'rl_reward_step', -0.02)
        self.rl_entropy_beta = getattr(self.global_config, 'rl_entropy_beta', 0.0)
        self.rl_grad_clip = getattr(self.global_config, 'rl_grad_clip', 0.0)
        self.rl_advantage_mode = getattr(
            self.global_config,
            'rl_advantage_mode',
            'episode_normalized',
        )
        self.rl_baseline_momentum = getattr(self.global_config, 'rl_baseline_momentum', 0.95)
        
        self.beam_width = getattr(self.global_config, 'beam_width', 25)    
        self.w_policy = getattr(self.global_config, 'w_policy', 0.2)
        self.normalize_rerank_features = getattr(self.global_config, 'normalize_rerank_features', False)
        self.normalize_path_by_length = getattr(self.global_config, 'normalize_path_by_length', False)
        self.rerank_fusion_mode = getattr(self.global_config, 'rerank_fusion_mode', 'score')
        self.rank_fusion_rrf_k = max(
            0,
            int(getattr(self.global_config, 'rank_fusion_rrf_k', 60)),
        )
        self.fact_head_protect_k = max(
            0,
            int(getattr(self.global_config, 'fact_head_protect_k', 0)),
        )
        self.similarity_head_protect_k = max(
            0,
            int(getattr(self.global_config, 'similarity_head_protect_k', 0)),
        )
        self.query_conditioned_ranker_path = str(getattr(
            self.global_config,
            'query_conditioned_ranker_path',
            '',
        )).strip()
        self.query_conditioned_fact_budget = max(1, int(getattr(
            self.global_config,
            'query_conditioned_fact_budget',
            200,
        )))
        self.query_conditioned_paragraph_budget = max(1, int(getattr(
            self.global_config,
            'query_conditioned_paragraph_budget',
            200,
        )))
        self.query_conditioned_repair_budget = max(0, int(getattr(
            self.global_config,
            'query_conditioned_repair_budget',
            50,
        )))
        self.query_conditioned_similarity_budget = max(0, int(getattr(
            self.global_config,
            'query_conditioned_similarity_budget',
            0,
        )))
        self.query_conditioned_ranker = None
        self.query_conditioned_ranker_mean = None
        self.query_conditioned_ranker_std = None
        self.query_conditioned_ranker_feature_selector = None
        self.query_conditioned_ranker_feature_schema = "legacy_v1"
        self.set_aware_selector_path = str(getattr(
            self.global_config,
            'set_aware_selector_path',
            '',
        )).strip()
        self.set_aware_selection_mode = str(getattr(
            self.global_config,
            'set_aware_selection_mode',
            'select2',
        )).lower()
        if self.set_aware_selection_mode not in {
                'select2', 'dynamic_setmax', 'dynamic_setmax_recovery'}:
            raise ValueError(
                f"Unsupported set_aware_selection_mode: {self.set_aware_selection_mode}"
            )
        self.set_aware_anchor_mode = str(getattr(
            self.global_config,
            'set_aware_anchor_mode',
            'unary',
        )).lower()
        if self.set_aware_anchor_mode not in {'unary', 'dpr'}:
            raise ValueError(
                f"Unsupported set_aware_anchor_mode: {self.set_aware_anchor_mode}"
            )
        self.set_aware_selector = None
        self.set_aware_selector_mean = None
        self.set_aware_selector_std = None
        self.set_aware_selector_mode = "residual"
        self.set_aware_uses_query_plan = False
        self.set_aware_uses_selection_progress = False
        self.set_aware_uses_dynamic_query_plan = False
        self.set_aware_bridge_candidate_limit = 0
        self.set_aware_openie_metadata = {}
        self.retrieval_source_mode = getattr(
            self.global_config,
            'retrieval_source_mode',
            'dual',
        )
        self.source_diagnostics = getattr(
            self.global_config,
            'source_diagnostics',
            False,
        )
        self.query_fact_score_mode = str(getattr(
            self.global_config,
            'query_fact_score_mode',
            'concat',
        )).lower()
        if self.query_fact_score_mode not in {'concat', 'slot_max', 'slot_mean', 'slot_hybrid'}:
            raise ValueError(
                f"Unsupported query_fact_score_mode: {self.query_fact_score_mode}"
            )
        self.query_slot_blend_alpha = float(np.clip(
            getattr(self.global_config, 'query_slot_blend_alpha', 0.5),
            0.0,
            1.0,
        ))
        self.last_source_diagnostics = {}
        self.w_fact_policy = getattr(self.global_config, 'w_fact_policy', 0.15)
        self.w_paragraph_policy = getattr(self.global_config, 'w_paragraph_policy', 0.15)
        self.w_sim = getattr(self.global_config, 'w_sim', 0.6)
        self.use_ppr_rerank = getattr(self.global_config, 'use_ppr_rerank', False)
        self.w_ppr = getattr(self.global_config, 'w_ppr', 0.0)
        self.w_repair_confidence = getattr(self.global_config, 'w_repair_confidence', 0.2)
        self.w_source_agreement = getattr(self.global_config, 'w_source_agreement', 0.05)
        self.use_learned_coordinator = getattr(self.global_config, 'use_learned_coordinator', False)
        self.coordinator_blend_alpha = getattr(self.global_config, 'coordinator_blend_alpha', 0.1)
        self.coordinator_protect_top_k = getattr(self.global_config, 'coordinator_protect_top_k', 0)
        self.coordinator_margin_gate_top_k = getattr(self.global_config, 'coordinator_margin_gate_top_k', 0)
        self.coordinator_margin_gate_threshold = getattr(self.global_config, 'coordinator_margin_gate_threshold', 0.0)
        self.coordinator_weights = None
        self.coordinator_bias = 0.0
        self.coordinator_mlp_state = None
        self.coordinator_feature_mean = None
        self.coordinator_feature_std = None
        
        self.w_query = float(np.clip(
            getattr(self.global_config, 'w_query', 0.5),
            0.0,
            1.0,
        ))
        self.cache_static_retrieval = bool(getattr(
            self.global_config,
            'cache_static_retrieval',
            True,
        ))
        self.optimize_static_graph_ops = bool(getattr(
            self.global_config, 'optimize_static_graph_ops', True,
        ))
        self.precompute_paragraph_knn = bool(getattr(
            self.global_config,
            'precompute_paragraph_knn',
            False,
        ))
        self.paragraph_knn_batch_size = max(1, int(getattr(
            self.global_config,
            'paragraph_knn_batch_size',
            256,
        )))
        self.batch_fact_beam_scoring = bool(getattr(
            self.global_config,
            'batch_fact_beam_scoring',
            True,
        ))
        self.reuse_fact_path_prefix = bool(getattr(
            self.global_config,
            'reuse_fact_path_prefix',
            True,
        ))
        self.batch_fact_path_mapping = bool(getattr(
            self.global_config, 'batch_fact_path_mapping', False,
        ))
        self.fact_traversal_mode = str(getattr(
            self.global_config,
            'fact_traversal_mode',
            'undirected',
        )).lower()
        if self.fact_traversal_mode not in {
                'undirected', 'hard_directed', 'role_aware'}:
            raise ValueError(
                f"Unsupported fact_traversal_mode: {self.fact_traversal_mode}"
            )
        self.fact_traversal_stats = {}
        self.fact_query_direction_cache = {}
        self.use_evidence_paragraph_graph = getattr(self.global_config, 'use_evidence_paragraph_graph', False)
        self.evidence_paragraph_max_candidates = getattr(self.global_config, 'evidence_paragraph_max_candidates', 80)
        self.evidence_paragraph_max_entity_support = getattr(self.global_config, 'evidence_paragraph_max_entity_support', 80)
        self.paragraph_seed_mode = getattr(
            self.global_config,
            'paragraph_seed_mode',
            'hybrid',
        )
        self.paragraph_fact_seed_score_mode = getattr(
            self.global_config,
            'paragraph_fact_seed_score_mode',
            'fixed',
        )
        self.last_paragraph_search_stats = {}
        
        self.rl_policy_network = None
        self.paragraph_policy_network = None
        self.rl_agent = None
        self.paragraph_agent = None
        self.policy_network_path = os.path.join(self.working_dir, "best_policy.pth")
        self.paragraph_policy_network_path = os.path.join(self.working_dir, "best_paragraph_policy.pth")

        requested_device = getattr(self.global_config, "execution_device", "cuda")
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA was requested but is unavailable; falling back to CPU.")
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        if self.rerank_fusion_mode == 'query_conditioned':
            self._load_query_conditioned_ranker()
            if self.set_aware_selector_path:
                self._load_set_aware_selector()
        
        logger.info(f"CoEviRAG initialization complete. RL Hyperparams: LR={self.rl_learning_rate}, Gamma={self.rl_gamma}")

        # Initialize query triplet cache
        query_cache_name = "query_triplets_cache.json"
        if str(self.global_config.llm_name).lower().startswith("google/gemma-4-"):
            model_slug = str(self.global_config.llm_name).replace("/", "_")
            query_cache_name = f"query_triplets_cache_{model_slug}.json"
        self.query_cache_path = os.path.join(self.global_config.save_dir, query_cache_name)
        self.query_triplets_cache = {}
        self.repair_cache_path = (
            str(getattr(self.global_config, "repair_cache_path", "")).strip()
            or os.path.join(self.global_config.save_dir, "repair_cache.json")
        )
        self.repair_cache = {}
        self.repair_stats = {
            "triggered": 0,
            "accepted": 0,
            "rejected": 0,
            "virtual_edges": 0,
            "relation_slots": 0,
            "resolved_slot_triples": 0,
            "unresolved_slots": 0,
            "verification_parse_failures": 0,
            "verification_retry_attempts": 0,
            "verification_retry_successes": 0,
            "trigger_examples": [],
            "structural_gate_checked": 0,
            "structural_gate_unparseable": 0,
            "required_slots": 0,
            "resolved_existing_slots": 0,
            "unresolved_required_slots": 0,
            "evidence_resolved_slots": 0,
            "retrieval_gap_slots": 0,
            "relation_evidence_matches": 0,
            "relation_evidence_misses": 0,
            "structural_gate_suppressed_queries": 0,
            "structural_gate_examples": [],
            "repair_passages": 0,
            "novel_repair_passages": 0,
            "gold_repair_passages": 0,
            "novel_gold_repair_passages": 0,
            "queries_with_novel_repair": 0,
            "queries_with_novel_gold_repair": 0,
            "overlay_anchored_triples": 0,
            "overlay_unanchored_triples": 0,
            "overlay_nodes": 0,
            "overlay_edges": 0,
            "overlay_linked_entities": 0,
            "overlay_expanded_nodes": 0,
            "overlay_expanded_passages": 0,
            "overlay_slot_guided_triples": 0,
            "semantic_expanded_passages": 0,
            "semantic_fallback_triples": 0,
            "overlay_unanchored_examples": [],
            "gold_hop_diagnostics": self._new_gold_hop_diagnostics(),
            "trigger_reasons": {},
            "coordinator_margin_gate_checked": 0,
            "coordinator_margin_gate_disabled": 0,
            "repair_gate_eligible": 0,
            "repair_gate_executed": 0,
            "repair_gate_skipped": 0,
        }
        self.runtime_node_embeddings = {}
        self.overlay_entity_link_cache = {}
        self.overlay_slot_embedding_cache = {}
        self.fact_node_to_chunk_ids = {}
        self.fact_node_to_triple = {}
        self.chunk_to_fact_node_ids = {}
        self.chunk_to_entity_node_ids = {}
        self.base_retrieval_masked_passages = set()
        self._repair_gold_hop_current = None
        self.repair_gate_model = None
        self.repair_gate_records = []
        self._repair_gate_features_current = None
        self._repair_gate_probability_current = None
        self._repair_gate_eligible_current = False
        self._repair_required_slots_current = []
        self.paragraph_neighbor_cache = {}
        self.paragraph_semantic_knn_cache = {}
        self.fact_neighbor_cache = {}
        self.fact_endpoint_cache = {}
        self.evidence_semantic_neighbor_cache = {}
        self.evidence_semantic_cache_signature = None
        self.fact_score_cache = {}
        self.fact_rank_cache = {}
        self.dense_retrieval_cache = {}
        self.static_retrieval_cache_stats = {
            "fact_score_hits": 0,
            "fact_score_misses": 0,
            "fact_rank_hits": 0,
            "fact_rank_misses": 0,
            "dense_hits": 0,
            "dense_misses": 0,
        }
        self.node_supporting_passages_cache = {}
        self.node_structural_score_cache = {}
        self._repair_normalized_passage_cache = {}
        self._repair_endpoint_fact_index = None
        
        if os.path.exists(self.query_cache_path):
            try:
                with open(self.query_cache_path, 'r', encoding='utf-8') as f:
                    self.query_triplets_cache = json.load(f)
                logger.info(f"Loaded {len(self.query_triplets_cache)} query triplet cache entries.")
            except Exception as e:
                logger.warning(f"Failed to load query cache: {e}")


    def load_rl_agent(self, training=False):
        """
        Loads the pre-trained policy network or initializes a new one for training.
        Also moves embeddings to the GPU for optimized retrieval performance.
        """
        if self.rl_agent is not None:
            return

        logger.info(f"Loading/Initializing policy network from {self.policy_network_path}...")
        
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
            
        # --- [Optimization] Move all Embeddings to GPU ---
        logger.info("Moving graph embeddings to GPU for training acceleration...")
        if self.entity_embeddings is not None:
            self.entity_embeddings_tensor = torch.from_numpy(self.entity_embeddings).float().to(self.device)
        if getattr(self, 'fact_embeddings', None) is not None:
            self.fact_embeddings_tensor = torch.from_numpy(self.fact_embeddings).float().to(self.device)
        if self.passage_embeddings is not None:
            self.passage_embeddings_tensor = torch.from_numpy(self.passage_embeddings).float().to(self.device)
        
        self.entity_key_to_idx = {k: i for i, k in enumerate(self.entity_node_keys)}
        self.fact_key_to_idx = {k: i for i, k in enumerate(self.fact_node_keys)}
        self.passage_key_to_idx = {k: i for i, k in enumerate(self.passage_node_keys)}

        if not hasattr(self.embedding_model, 'embedding_dim') or self.embedding_model.embedding_dim is None:
             if self.passage_embeddings is not None and self.passage_embeddings.shape[1] > 0:
                 embedding_dim = self.passage_embeddings.shape[1]
             else:
                raise ValueError("Cannot determine embedding dimension. Ensure embedding_model has 'embedding_dim' or embeddings are loaded.")
        else:
            embedding_dim = self.embedding_model.embedding_dim
            
        logger.info(f"Dynamically retrieved embedding dimension: {embedding_dim}")

        self.rl_policy_network = PolicyNetwork(embedding_dim, self.rl_hidden_dim)
        self.paragraph_policy_network = PolicyNetwork(embedding_dim, self.rl_hidden_dim)
        
        if os.path.exists(self.policy_network_path):
            fact_load = self.rl_policy_network.load_state_dict(
                torch.load(self.policy_network_path, map_location=self.device),
                strict=False,
            )
            if fact_load.missing_keys:
                logger.info("Initialized new Fact policy parameters: %s", fact_load.missing_keys)
            logger.info("Policy model loaded successfully.")
        elif not training:
            logger.error("No pre-trained policy model found.")
            raise FileNotFoundError(f"Policy file not found at {self.policy_network_path}")

        if os.path.exists(self.paragraph_policy_network_path):
            paragraph_load = self.paragraph_policy_network.load_state_dict(
                torch.load(self.paragraph_policy_network_path, map_location=self.device),
                strict=False,
            )
            if paragraph_load.missing_keys:
                logger.info("Initialized new Paragraph policy parameters: %s", paragraph_load.missing_keys)
            logger.info("Paragraph policy model loaded successfully.")
        else:
            self.paragraph_policy_network.load_state_dict(self.rl_policy_network.state_dict())
            logger.info("Paragraph policy initialized from fact policy.")
        

        self.rl_policy_network.to(self.device)
        self.paragraph_policy_network.to(self.device)
        self.rl_agent = RLAgent(
            self.rl_policy_network,
            self.rl_learning_rate,
            self.rl_gamma,
            entropy_beta=self.rl_entropy_beta,
            grad_clip=self.rl_grad_clip,
            advantage_mode=self.rl_advantage_mode,
            baseline_momentum=self.rl_baseline_momentum,
        )
        self.paragraph_agent = RLAgent(
            self.paragraph_policy_network,
            self.rl_learning_rate,
            self.rl_gamma,
            entropy_beta=self.rl_entropy_beta,
            grad_clip=self.rl_grad_clip,
            advantage_mode=self.rl_advantage_mode,
            baseline_momentum=self.rl_baseline_momentum,
        )
        self.rl_policy_network.train() if training else self.rl_policy_network.eval()
        self.paragraph_policy_network.train() if training else self.paragraph_policy_network.eval()

    def get_embedding_for_node(self, node_key: str) -> Optional[torch.Tensor]:
        """
        Retrieves the corresponding embedding tensor for a given node key (entity or passage) from the GPU cache.
        """
        if hasattr(self, 'entity_embeddings_tensor'):
            idx = self.entity_key_to_idx.get(node_key)
            if idx is not None:
                return self.entity_embeddings_tensor[idx]

        if hasattr(self, 'fact_embeddings_tensor'):
            idx = self.fact_key_to_idx.get(node_key)
            if idx is not None:
                return self.fact_embeddings_tensor[idx]

        if node_key in self.runtime_node_embeddings:
            return self.runtime_node_embeddings[node_key]
        
        if hasattr(self, 'passage_embeddings_tensor'):
            idx = self.passage_key_to_idx.get(node_key)
            if idx is not None:
                return self.passage_embeddings_tensor[idx]
        return None

    def _get_fact_neighbors(self, node_key: str) -> Tuple[str, ...]:
        """Return immutable base-KG neighbor names without repeated igraph lookups."""
        cached = self.fact_neighbor_cache.get(node_key)
        if cached is not None:
            return cached
        node_idx = getattr(self, "node_name_to_vertex_idx", {}).get(node_key)
        if node_idx is None:
            return ()
        neighbors = tuple(
            self.graph.vs[idx]["name"] for idx in self.graph.neighbors(node_idx)
        )
        self.fact_neighbor_cache[node_key] = neighbors
        return neighbors

    def _fact_endpoint_keys(self, fact_key: str) -> Optional[Tuple[str, str]]:
        """Return the hashed head/tail entity keys for an OpenIE fact node."""
        triple = getattr(self, "fact_node_to_triple", {}).get(fact_key)
        if not triple or len(triple) != 3:
            return None
        if not getattr(self, "optimize_static_graph_ops", False):
            return (
                compute_mdhash_id(content=str(triple[0]).lower(), prefix="entity-"),
                compute_mdhash_id(content=str(triple[2]).lower(), prefix="entity-"),
            )
        # Atomic string entries avoid retaining thousands of GC-tracked tuple keys.
        endpoints = []
        for value in (triple[0], triple[2]):
            text = str(value).lower()
            key = self.fact_endpoint_cache.get(text)
            if key is None:
                key = compute_mdhash_id(content=text, prefix="entity-")
                self.fact_endpoint_cache[text] = key
            endpoints.append(key)
        return tuple(endpoints)

    @staticmethod
    def _query_slot_value_is_known(value: Any) -> bool:
        normalized = str(value).strip().lower()
        return normalized not in {"", "?", "unknown", "none", "null"}

    def _query_direction_for_fact(
            self, query: str, fact_key: str) -> Optional[str]:
        """Infer whether a query slot needs the fact's forward or reverse role."""
        cache_key = (query, fact_key)
        if cache_key in self.fact_query_direction_cache:
            return self.fact_query_direction_cache[cache_key]
        triple = getattr(self, "fact_node_to_triple", {}).get(fact_key)
        if not triple or len(triple) != 3:
            return None
        fact_family = self._repair_relation_family(triple[1], tail=triple[2])
        head_aliases = self._repair_entity_aliases(triple[0])
        tail_aliases = self._repair_entity_aliases(triple[2])
        directions = set()
        for slot in self._get_raw_query_triples(query):
            query_head, query_relation, query_tail = slot[:3]
            query_family = self._repair_relation_family(
                query_relation,
                query=query,
                tail=query_tail,
            )
            if query_family != fact_family:
                continue
            head_known = self._query_slot_value_is_known(query_head)
            tail_known = self._query_slot_value_is_known(query_tail)
            # Only incomplete slots provide an unambiguous navigation anchor.
            if head_known == tail_known:
                continue
            anchor = query_head if head_known else query_tail
            anchor_aliases = self._repair_entity_aliases(anchor)
            if anchor_aliases & head_aliases:
                directions.add("forward")
            if anchor_aliases & tail_aliases:
                directions.add("reverse")
        direction = next(iter(directions)) if len(directions) == 1 else None
        self.fact_query_direction_cache[cache_key] = direction
        return direction

    def _get_fact_search_neighbors(
            self,
            node_key: str,
            query: str,
            previous_node: Optional[str] = None) -> Tuple[str, ...]:
        """Apply hard or query-role-aware direction constraints to Fact traversal."""
        neighbors = self._get_fact_neighbors(node_key)
        mode = self.fact_traversal_mode
        stats = self.fact_traversal_stats
        stats["calls"] = stats.get("calls", 0) + 1
        stats["neighbors_before"] = stats.get("neighbors_before", 0) + len(neighbors)
        if mode == "undirected" or not neighbors:
            stats["neighbors_after"] = stats.get("neighbors_after", 0) + len(neighbors)
            if not neighbors:
                stats["dead_ends"] = stats.get("dead_ends", 0) + 1
            return neighbors

        kept = []
        if node_key.startswith("fact-"):
            endpoints = self._fact_endpoint_keys(node_key)
            if endpoints is None:
                return neighbors
            head_key, tail_key = endpoints
            if mode == "hard_directed":
                kept = [neighbor for neighbor in neighbors if neighbor == tail_key]
            elif previous_node == head_key:
                kept = [neighbor for neighbor in neighbors if neighbor == tail_key]
            elif previous_node == tail_key:
                kept = [neighbor for neighbor in neighbors if neighbor == head_key]
            else:
                direction = self._query_direction_for_fact(query, node_key)
                target = tail_key if direction == "forward" else head_key
                kept = (
                    [neighbor for neighbor in neighbors if neighbor == target]
                    if direction is not None else list(neighbors)
                )
        else:
            for neighbor in neighbors:
                if not neighbor.startswith("fact-"):
                    # Synonym edges are semantically symmetric in every mode.
                    kept.append(neighbor)
                    continue
                endpoints = self._fact_endpoint_keys(neighbor)
                if endpoints is None:
                    kept.append(neighbor)
                    continue
                head_key, tail_key = endpoints
                if mode == "hard_directed":
                    if node_key == head_key:
                        kept.append(neighbor)
                    continue
                direction = self._query_direction_for_fact(query, neighbor)
                if (
                    direction is None
                    or (direction == "forward" and node_key == head_key)
                    or (direction == "reverse" and node_key == tail_key)
                ):
                    kept.append(neighbor)

        stats["neighbors_after"] = stats.get("neighbors_after", 0) + len(kept)
        if not kept:
            stats["dead_ends"] = stats.get("dead_ends", 0) + 1
        return tuple(kept)
    
    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5) -> np.ndarray:
        """
        Runs Personalized PageRank (PPR) on the knowledge graph using the provided reset probability vector.
        Returns the raw probability array.
        """
        if damping is None: damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            damping=damping, directed=False, weights='weight', reset=reset_prob, implementation='prpack'
        )
        return np.array(pagerank_scores)

    def extract_query_triplets(self, query: str, persist_cache: bool = True) -> str:
        """
        Extracts structured triplets (S, R, O) from a query using OpenIE and caches the result.
        """
        # 1. Check cache
        if query in self.query_triplets_cache:
            cache_item = self.query_triplets_cache[query]
            if isinstance(cache_item, dict):
                return cache_item.get("extracted_triplets", "")
            return cache_item

        # 2. Initialize OpenIE
        if not hasattr(self, 'query_openie'):
            try:
                self.query_openie = OpenIE(llm_model=self.llm_model)
            except Exception as e:
                logger.warning(f"OpenIE initialization failed: {e}")
                return ""

        try:
            chunk_key = compute_mdhash_id(content=query, prefix="query_")
            
            # 3. Execute extraction (NER + Triple Extraction)
            ner_output = self.query_openie.ner(chunk_key=chunk_key, passage=query)
            if ner_output.metadata.get("error"):
                logger.warning(f"Query NER failed; not caching empty query triples for: {query[:80]}")
                return ""
            
            triple_output = self.query_openie.triple_extraction(
                chunk_key=chunk_key, 
                passage=query, 
                named_entities=ner_output.unique_entities,
                prompt_name='query_triple_extraction' 
            )
            if triple_output.metadata.get("error"):
                logger.warning(f"Query triple extraction failed; not caching empty triples for: {query[:80]}")
                return ""
            
            triples = triple_output.triples
            
            # 4. Handle empty results
            if not triples:
                logger.info(f"No query triples extracted; using raw query only for: {query[:80]}")
                return ""
                
            # 5. Format output
            triplet_strings = []
            for t in triples:
                if len(t) < 3: continue
                s, r, o = t[0], t[1], t[2]
                
                parts = []
                if s != "?": parts.append(s)
                if r != "?": parts.append(r)
                if o != "?": parts.append(o)
                
                phrase = " ".join(parts)
                triplet_strings.append(phrase)
            
            result_str = " ".join(triplet_strings)
            
            # 6. Build and save rich cache structure
            cache_entry = {
                "original_query": query,
                "extracted_triplets": result_str,
                "raw_triples": triples 
            }

            self.query_triplets_cache[query] = cache_entry
            
            if persist_cache:
                try:
                    with open(self.query_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(self.query_triplets_cache, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Failed to write query cache: {e}")

            return result_str
            
        except Exception as e:
            logger.warning(f"Exception during query triplet extraction: {e}")
            return ""
            
    def _get_query_slot_strings(self, query: str) -> List[str]:
        slot_strings = []
        seen = set()
        for triple in self._get_raw_query_triples(query):
            slot = " ".join(
                str(part).strip()
                for part in triple[:3]
                if str(part).strip() and str(part).strip() != "?"
            )
            if slot and slot not in seen:
                seen.add(slot)
                slot_strings.append(slot)
        return slot_strings

    def precompute_embeddings(self, queries: List[str]):
        """
        Pre-computes embeddings for raw queries, extracted triplets, and DPR paths in batch to save time during retrieval.
        """
        logger.info(f"Starting pre-computation for {len(queries)} query embeddings...")
        
        # --- 1. Raw Query Encoding (Path A) ---
        raw_queries_to_encode = [q for q in queries if q not in self.query_to_embedding['triple']]
        if raw_queries_to_encode:
            logger.info(f"Encoding {len(raw_queries_to_encode)} raw queries...")
            raw_embeddings = self.embedding_model.batch_encode(
                raw_queries_to_encode,
                instruction=get_query_instruction('query_to_fact'),
                norm=True
            )
            for q, emb in zip(raw_queries_to_encode, raw_embeddings):
                self.query_to_embedding['triple'][q] = emb

        if self.w_query >= 1.0:
            passage_queries = [
                q for q in queries
                if q not in self.query_to_embedding['passage']
            ]
            if passage_queries:
                logger.info(f"Encoding {len(passage_queries)} queries for DPR...")
                passage_embeddings = self.embedding_model.batch_encode(
                    passage_queries,
                    instruction=get_query_instruction('query_to_passage'),
                    norm=True,
                )
                for q, emb in zip(passage_queries, passage_embeddings):
                    self.query_to_embedding['passage'][q] = emb
            logger.info("All query-only embedding pre-computations finished!")
            return
        
        # --- 2. [Optimization] Concurrent Triplet Extraction (Path B) ---
        triplets_to_encode = []
        queries_needing_extraction = [q for q in queries if q not in self.query_triplets_cache]
        
        if queries_needing_extraction:
            logger.info(f"Extracting triplets for {len(queries_needing_extraction)} queries...")
            
            def extract_worker(q):
                return q, self.extract_query_triplets(q, persist_cache=False)

            max_workers = getattr(self.global_config, "query_extraction_max_workers", 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(extract_worker, q): q for q in queries_needing_extraction}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting query triplets"):
                    q, t_str = future.result()    

            # Bound preprocessing latency: failed or legitimately empty OpenIE
            # outputs use the raw-query path instead of being retried serially
            # by the collection loop below.
            fallback_queries = [
                q for q in queries_needing_extraction
                if q not in self.query_triplets_cache
            ]
            for q in fallback_queries:
                self.query_triplets_cache[q] = {
                    "original_query": q,
                    "extracted_triplets": "",
                    "raw_triples": [],
                    "fallback_reason": "query_openie_empty_or_failed",
                }
            if fallback_queries:
                logger.warning(
                    "Using raw-query fallback for %d query OpenIE failures; "
                    "skipping unbounded serial retries.",
                    len(fallback_queries),
                )
                    
        try:
            with open(self.query_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.query_triplets_cache, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        # --- 3. Collect Triplets and Encode ---
        for q in queries:
            triplets_str = self.extract_query_triplets(q)
            if triplets_str and triplets_str not in self.query_to_embedding['triple']:
                triplets_to_encode.append(triplets_str)
        
        unique_triplets = sorted(set(triplets_to_encode))
        if unique_triplets:
            logger.info(f"Encoding {len(unique_triplets)} unique triplet strings...")
            triplet_embeddings = self.embedding_model.batch_encode(
                unique_triplets,
                instruction=get_query_instruction('fact_to_fact'),
                norm=True
            )
            for t_str, emb in zip(unique_triplets, triplet_embeddings):
                self.query_to_embedding['triple'][t_str] = emb

        if self.query_fact_score_mode != 'concat':
            slot_strings = {
                slot
                for query in queries
                for slot in self._get_query_slot_strings(query)
                if slot not in self.query_to_embedding['triple']
            }
            unique_slots = sorted(slot_strings)
            if unique_slots:
                logger.info(
                    "Encoding %d unique structured query slots for %s scoring...",
                    len(unique_slots),
                    self.query_fact_score_mode,
                )
                slot_embeddings = self.embedding_model.batch_encode(
                    unique_slots,
                    instruction=get_query_instruction('fact_to_fact'),
                    norm=True,
                )
                for slot, emb in zip(unique_slots, slot_embeddings):
                    self.query_to_embedding['triple'][slot] = emb

        # --- 4. DPR Encoding (Path C) ---
        passage_queries = [q for q in queries if q not in self.query_to_embedding['passage']]
        if passage_queries:
            logger.info(f"Encoding {len(passage_queries)} queries for DPR...")
            p_embeddings = self.embedding_model.batch_encode(
                passage_queries,
                instruction=get_query_instruction('query_to_passage'),
                norm=True
            )
            for q, emb in zip(passage_queries, p_embeddings):
                self.query_to_embedding['passage'][q] = emb

        logger.info("All embedding pre-computations finished!")

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Calculates fusion scores using dual-path recall: raw query similarity and triplet-based similarity.
        """
        if self.cache_static_retrieval:
            cached_scores = self.fact_score_cache.get(query)
            if cached_scores is not None:
                self.static_retrieval_cache_stats["fact_score_hits"] += 1
                return cached_scores
            self.static_retrieval_cache_stats["fact_score_misses"] += 1

        def cache_scores(scores: np.ndarray) -> np.ndarray:
            if self.cache_static_retrieval:
                self.fact_score_cache[query] = scores
            return scores

        # --- Path A: Raw Query ---
        query_emb_A = self.query_to_embedding['triple'].get(query, None)
        if query_emb_A is None:
            query_emb_A = self.embedding_model.batch_encode(query, 
                                                            instruction=get_query_instruction('query_to_fact'), 
                                                            norm=True)
            self.query_to_embedding['triple'][query] = query_emb_A

        if len(self.fact_embeddings) == 0:
            return np.array([])

        scores_A = np.dot(self.fact_embeddings, query_emb_A.T)
        scores_A = np.squeeze(scores_A) if scores_A.ndim == 2 else scores_A
        scores_A = min_max_normalize(scores_A)

        if self.w_query >= 1.0:
            return cache_scores(scores_A)

        # --- Path B: Structured Triplets ---
        triplets_str = self.extract_query_triplets(query)
        
        if not triplets_str:
            return cache_scores(scores_A)
            
        if self.query_fact_score_mode == 'concat':
            query_emb_B = self.query_to_embedding['triple'].get(triplets_str, None)
            if query_emb_B is None:
                query_emb_B = self.embedding_model.batch_encode(
                    triplets_str,
                    instruction=get_query_instruction('fact_to_fact'),
                    norm=True,
                )
                self.query_to_embedding['triple'][triplets_str] = query_emb_B

            scores_B = np.dot(self.fact_embeddings, query_emb_B.T)
            scores_B = np.squeeze(scores_B) if scores_B.ndim == 2 else scores_B
            scores_B = min_max_normalize(scores_B)
        else:
            slot_strings = self._get_query_slot_strings(query)
            if not slot_strings:
                return scores_A
            slot_embeddings = []
            for slot in slot_strings:
                slot_emb = self.query_to_embedding['triple'].get(slot)
                if slot_emb is None:
                    slot_emb = self.embedding_model.batch_encode(
                        slot,
                        instruction=get_query_instruction('fact_to_fact'),
                        norm=True,
                    )
                    self.query_to_embedding['triple'][slot] = slot_emb
                slot_embeddings.append(np.asarray(slot_emb).reshape(-1))

            slot_score_matrix = np.dot(
                self.fact_embeddings,
                np.stack(slot_embeddings, axis=1),
            )
            normalized_slots = np.stack(
                [min_max_normalize(slot_score_matrix[:, idx])
                 for idx in range(slot_score_matrix.shape[1])],
                axis=1,
            )
            if self.query_fact_score_mode in {'slot_max', 'slot_hybrid'}:
                scores_B = normalized_slots.max(axis=1)
            else:
                scores_B = normalized_slots.mean(axis=1)

            if self.query_fact_score_mode == 'slot_hybrid':
                concat_emb = self.query_to_embedding['triple'].get(triplets_str)
                if concat_emb is None:
                    concat_emb = self.embedding_model.batch_encode(
                        triplets_str,
                        instruction=get_query_instruction('fact_to_fact'),
                        norm=True,
                    )
                    self.query_to_embedding['triple'][triplets_str] = concat_emb
                concat_scores = np.dot(self.fact_embeddings, concat_emb.T)
                concat_scores = (
                    np.squeeze(concat_scores)
                    if concat_scores.ndim == 2
                    else concat_scores
                )
                concat_scores = min_max_normalize(concat_scores)
                alpha = self.query_slot_blend_alpha
                scores_B = (1.0 - alpha) * concat_scores + alpha * scores_B
        
        w_query = self.w_query 
        final_scores = w_query * scores_A + (1 - w_query) * scores_B
        
        return cache_scores(final_scores)
            
    def _get_ppr_inputs_and_scores(self, query, top_k_facts, top_k_fact_indices):
        """
        Helper method to calculate node weights for PPR and the resulting PPR scores for all nodes.
        """
        passage_node_weight = getattr(self.global_config, 'passage_node_weight', 0.05)
        
        linking_score_map, phrase_scores = {}, {}
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        query_fact_scores = self.get_fact_scores(query)
        for rank, f in enumerate(top_k_facts):
            subject_phrase, object_phrase = f[0].lower(), f[2].lower()
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                if phrase_key in self.node_name_to_vertex_idx:
                    phrase_id = self.node_name_to_vertex_idx[phrase_key]
                    current_score = phrase_weights[phrase_id]
                    new_score = fact_score
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        new_score /= len(self.ent_node_to_chunk_ids[phrase_key])
                    phrase_weights[phrase_id] = max(current_score, new_score) 
                if phrase not in phrase_scores: phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)
        
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))
        
        if getattr(self.global_config, 'linking_top_k', None):
            phrase_weights, _ = self.get_top_k_weights(self.global_config.linking_top_k, phrase_weights, linking_score_map)

        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)
        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            if passage_node_key in self.node_name_to_vertex_idx:
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
                passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight

        node_weights = phrase_weights + passage_weights
        
        all_node_pagerank_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        all_node_names = self.graph.vs["name"]
        ppr_scores_dict = {name: score for name, score in zip(all_node_names, all_node_pagerank_scores)}
        
        return node_weights, ppr_scores_dict

    def _fact_to_node_key(self, fact: Tuple) -> Optional[str]:
        try:
            candidates = [
                compute_mdhash_id(content=str(tuple(fact)), prefix="fact-"),
                compute_mdhash_id(content=str(tuple(text_processing(list(fact)))), prefix="fact-"),
            ]
        except Exception:
            return None

        for fact_key in candidates:
            if fact_key in getattr(self, "node_name_to_vertex_idx", {}):
                return fact_key
        return None

    def get_ranked_seed_nodes(
            self,
            top_k_facts: List[Tuple],
            top_k_fact_indices: Optional[List[int]] = None,
            query_fact_scores: Optional[np.ndarray] = None,
            max_seeds: Optional[int] = None,
            include_fact_nodes: bool = True) -> List[Tuple[str, float]]:
        """Rank graph start nodes with fact score, support size, and retrieval rank."""
        seed_scores = {}

        def update_seed(node_key: str, score: float) -> None:
            if node_key not in getattr(self, "node_name_to_vertex_idx", {}):
                return
            if self.get_embedding_for_node(node_key) is None:
                return
            seed_scores[node_key] = max(seed_scores.get(node_key, -float("inf")), float(score))

        for rank, fact in enumerate(top_k_facts or []):
            if query_fact_scores is not None and top_k_fact_indices is not None and rank < len(top_k_fact_indices):
                fact_score = float(query_fact_scores[top_k_fact_indices[rank]])
            else:
                fact_score = 1.0 / float(rank + 1)
            rank_bonus = 1.0 / float(rank + 1)

            if include_fact_nodes:
                fact_key = self._fact_to_node_key(fact)
                if fact_key:
                    fact_support = len(getattr(self, "fact_node_to_chunk_ids", {}).get(fact_key, set()))
                    update_seed(fact_key, fact_score + 0.10 * rank_bonus + 0.02 * np.log1p(fact_support))

            for entity in [fact[0], fact[2]]:
                entity_key = compute_mdhash_id(content=str(entity).lower(), prefix="entity-")
                entity_support = len(getattr(self, "ent_node_to_chunk_ids", {}).get(entity_key, set()))
                update_seed(entity_key, fact_score + 0.05 * rank_bonus + 0.02 * np.log1p(entity_support))

        ranked = sorted(
            seed_scores.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        if max_seeds is not None and max_seeds > 0:
            ranked = ranked[:max_seeds]
        return ranked

    def retrieve(self, queries: List[str], num_to_retrieve: int = None, gold_docs: List[List[str]] = None) -> Union[List[QuerySolution], Tuple[List[QuerySolution], Dict]]:
            """
            Retrieves relevant passages for a set of queries using strictly the 'rl_rerank' mode.
            """
            # Always load the RL agent since we are strictly in rl_rerank mode
            if self.rl_agent is None:
                self.load_rl_agent(training=False)
    
            if num_to_retrieve is None: num_to_retrieve = self.global_config.retrieval_top_k
            if gold_docs is not None: retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
            if not self.ready_to_retrieve: self.prepare_retrieval_objects()
            self._reset_repair_stats()
            self.fact_traversal_stats = {
                "mode": self.fact_traversal_mode,
                "calls": 0,
                "neighbors_before": 0,
                "neighbors_after": 0,
                "dead_ends": 0,
            }
    
            self.precompute_embeddings(queries)
            
            retrieve_start_time = time.time()
            retrieval_results = []
            triggered_query_indices = []
            self.last_repair_triggered_query_indices = []
            self.last_paragraph_search_stats = {
                "queries": 0,
                "fact_seed_candidates": 0,
                "dpr_seed_candidates": 0,
                "seed_overlap": 0,
                "selected_fact_seeds": 0,
                "selected_dpr_seeds": 0,
                "selected_total_seeds": 0,
            }
            self.last_fact_seed_gate_stats = {
                "checked": 0,
                "kept": 0,
                "skipped": 0,
                "keep_probability_sum": 0.0,
            }
            source_diagnostic_docs = (
                {"fact": [], "paragraph": [], "union_similarity": []}
                if gold_docs is not None and self.source_diagnostics
                else None
            )
            desc = "Retrieving with CoEviRAG"
    
            for q_idx, query in tqdm(enumerate(queries), desc=desc, total=len(queries)):
                self._repair_triggered_current_query = False
                self._repair_required_slots_current = []
                self._repair_gate_features_current = None
                self._repair_gate_probability_current = None
                self._repair_gate_eligible_current = False
                novel_gold_repair_count = 0
                    
                query_fact_scores = self.get_fact_scores(query)
                
                top_k_fact_indices, top_k_facts = self.rank_facts(query, query_fact_scores)
    
                query_emb_tensor = self._get_query_embedding_tensor(query)
                dpr_ids_for_repair, dpr_scores_for_repair = self.dense_passage_retrieval(query)

                if top_k_facts:
                    fact_info, query_emb_raw = self.run_rl_graph_search_rerank(
                        query,
                        top_k_facts,
                        top_k_fact_indices=top_k_fact_indices,
                        query_fact_scores=query_fact_scores
                    )
                    if isinstance(query_emb_raw, np.ndarray):
                        query_emb_tensor = torch.from_numpy(query_emb_raw).float().to(self.device).view(1, -1)
                    else:
                        query_emb_tensor = query_emb_raw.to(self.device).view(1, -1)
                else:
                    fact_info = {}

                ranked_fact_seed_items = sorted(
                        fact_info.items(),
                        key=lambda item: (
                            item[1].get("score", -float("inf")),
                            item[0],
                        ),
                        reverse=True
                    )[:getattr(self.global_config, "paragraph_agent_seed_top_k", 20)]
                ranked_fact_seed_items = self._gate_fact_seeds_for_paragraph(
                    ranked_fact_seed_items,
                    query_emb_tensor,
                )
                fact_seed_scores = (
                    min_max_normalize(np.array([
                        float(info.get("score", 0.0))
                        for _, info in ranked_fact_seed_items
                    ]))
                    if ranked_fact_seed_items
                    else np.array([], dtype=np.float32)
                )
                fact_seed_passages = [
                    (passage_key, float(seed_score))
                    for (passage_key, _), seed_score in zip(
                        ranked_fact_seed_items,
                        fact_seed_scores,
                    )
                ]
                if getattr(self.global_config, "use_paragraph_agent", True):
                    paragraph_info = self.run_paragraph_graph_search(
                        query=query,
                        seed_passages=fact_seed_passages,
                        query_emb=query_emb_tensor
                    )
                else:
                    paragraph_info = {}

                self._prepare_gold_hop_diagnostic(
                    gold_contents=gold_docs[q_idx] if gold_docs is not None and q_idx < len(gold_docs) else None,
                    base_passages=set(fact_info).union(paragraph_info),
                )
                repair_info = {}
                if self._should_trigger_repair(
                        query,
                        fact_info,
                        paragraph_info=paragraph_info,
                        dpr_ids=dpr_ids_for_repair):
                    active_paths = [
                        info.get("path", [])
                        for _, info in sorted(
                            fact_info.items(),
                            key=lambda item: (
                                item[1].get("score", -float("inf")),
                                item[0],
                            ),
                            reverse=True
                        )[:3]
                    ]
                    repair_info = self._repair_broken_path(
                        query=query,
                        top_k_facts=top_k_facts,
                        active_paths=active_paths
                    )
                if self._repair_triggered_current_query:
                    self._finalize_gold_hop_diagnostic(query)

                if repair_info:
                    repair_passages = set(repair_info)
                    base_passages = set(fact_info).union(paragraph_info)
                    novel_repair_passages = repair_passages - base_passages
                    self.repair_stats["repair_passages"] += len(repair_passages)
                    self.repair_stats["novel_repair_passages"] += len(novel_repair_passages)
                    if novel_repair_passages:
                        self.repair_stats["queries_with_novel_repair"] += 1

                    if gold_docs is not None and q_idx < len(gold_docs):
                        gold_set = set(gold_docs[q_idx])

                        def is_gold_passage(passage_key: str) -> bool:
                            try:
                                content = self.chunk_embedding_store.get_row(passage_key)["content"]
                            except Exception:
                                return False
                            return content in gold_set

                        gold_repair = {key for key in repair_passages if is_gold_passage(key)}
                        novel_gold_repair = gold_repair.intersection(novel_repair_passages)
                        novel_gold_repair_count = len(novel_gold_repair)
                        self.repair_stats["gold_repair_passages"] += len(gold_repair)
                        self.repair_stats["novel_gold_repair_passages"] += len(novel_gold_repair)
                        if novel_gold_repair:
                            self.repair_stats["queries_with_novel_gold_repair"] += 1

                all_passage_info = self._merge_agent_passage_info(
                    fact_info,
                    paragraph_info,
                    repair_info,
                )
                if self.retrieval_source_mode == "fact_only":
                    passage_info = self._merge_agent_passage_info(
                        fact_info,
                        {},
                        repair_info,
                    )
                elif self.retrieval_source_mode == "paragraph_only":
                    passage_info = self._merge_agent_passage_info(
                        {},
                        paragraph_info,
                        repair_info,
                    )
                else:
                    passage_info = all_passage_info

                if (
                        self.rerank_fusion_mode == "query_conditioned"
                        and self.query_conditioned_similarity_budget > 0):
                    passage_info = dict(passage_info)
                    for passage_idx, dense_score in zip(
                            dpr_ids_for_repair[:self.query_conditioned_similarity_budget],
                            dpr_scores_for_repair[:self.query_conditioned_similarity_budget]):
                        passage_key = self.passage_node_keys[int(passage_idx)]
                        if passage_key not in passage_info:
                            passage_info[passage_key] = {
                                "score": float(dense_score),
                                "path": [],
                                "paths": [],
                                "sources": [],
                                "source_scores": {},
                                "source_path_lengths": {},
                                "source_details": {},
                            }

                if not passage_info:
                    logger.warning("Dual-agent retrieval found no passages, falling back to DPR.")
                    sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                else:
                    ppr_scores_dict = {}
                    if self.use_ppr_rerank and self.w_ppr > 0 and top_k_facts:
                        _, ppr_scores_dict = self._get_ppr_inputs_and_scores(query, top_k_facts, top_k_fact_indices)

                    candidate_components = self._build_candidate_components(
                        passage_info=passage_info,
                        query_emb_tensor=query_emb_tensor,
                        ppr_scores_dict=ppr_scores_dict,
                    )
                    if source_diagnostic_docs is not None:
                        diagnostic_components = (
                            candidate_components
                            if passage_info is all_passage_info
                            else self._build_candidate_components(
                                passage_info=all_passage_info,
                                query_emb_tensor=query_emb_tensor,
                                ppr_scores_dict=ppr_scores_dict,
                            )
                        )
                        diagnostic_rankings = {
                            "fact": [
                                record["passage"] for record in sorted(
                                    (
                                        item for item in diagnostic_components
                                        if item["fact"] is not None
                                    ),
                                    key=lambda item: (
                                        item["fact"],
                                        item["passage"],
                                    ),
                                    reverse=True,
                                )
                            ],
                            "paragraph": [
                                record["passage"] for record in sorted(
                                    (
                                        item for item in diagnostic_components
                                        if item["paragraph"] is not None
                                    ),
                                    key=lambda item: (
                                        item["paragraph"],
                                        item["passage"],
                                    ),
                                    reverse=True,
                                )
                            ],
                            "union_similarity": [
                                record["passage"] for record in sorted(
                                    diagnostic_components,
                                    key=lambda item: (
                                        item["similarity"],
                                        item["passage"],
                                    ),
                                    reverse=True,
                                )
                            ],
                        }
                        for source_name, passage_keys in diagnostic_rankings.items():
                            source_diagnostic_docs[source_name].append([
                                self.chunk_embedding_store.get_row(key)["content"]
                                for key in passage_keys
                            ])
                    normalized_components = self._normalize_candidate_components(candidate_components)
                    candidate_records = []
                    if self.rerank_fusion_mode == "query_conditioned":
                        ranker_scores = self._query_conditioned_candidate_scores(
                            candidate_components
                        )
                        for record in candidate_components:
                            passage = record["passage"]
                            if passage in ranker_scores:
                                candidate_records.append((
                                    passage,
                                    record["info"],
                                    ranker_scores[passage],
                                ))
                    elif self.rerank_fusion_mode == "rank_rrf":
                        rank_fusion_scores = self._rank_fusion_candidate_scores(
                            candidate_components
                        )
                        for record in candidate_components:
                            passage = record["passage"]
                            candidate_records.append(
                                (
                                    passage,
                                    record["info"],
                                    rank_fusion_scores[passage],
                                )
                            )
                    elif self.rerank_fusion_mode == "similarity":
                        for record in candidate_components:
                            candidate_records.append(
                                (
                                    record["passage"],
                                    record["info"],
                                    record["similarity"],
                                )
                            )
                    elif self.normalize_rerank_features:
                        for record in candidate_components:
                            passage = record["passage"]
                            heuristic_score = (
                                self.w_fact_policy * normalized_components["fact"][passage]
                                + self.w_paragraph_policy * normalized_components["paragraph"][passage]
                                + self.w_sim * normalized_components["similarity"][passage]
                                + self.w_ppr * normalized_components["ppr"][passage]
                                + self.w_repair_confidence * record["repair"]
                                + self.w_source_agreement * min(1.0, record["agreement"])
                            )
                            candidate_records.append((passage, record["info"], heuristic_score))
                    else:
                        for record in candidate_components:
                            heuristic_score = (
                                self.w_policy * record["path"]
                                + self.w_sim * record["similarity"]
                                + self.w_ppr * record["ppr"]
                                + self.w_repair_confidence * record["repair"]
                                + self.w_source_agreement * record["agreement"]
                            )
                            candidate_records.append((record["passage"], record["info"], heuristic_score))

                    protected_passages = set()
                    if (
                            self.rerank_fusion_mode != "query_conditioned"
                            and self.coordinator_protect_top_k > 0):
                        protected_passages = {
                            passage for passage, _, _ in sorted(
                                candidate_records,
                                key=lambda item: (item[2], item[0]),
                                reverse=True
                            )[:self.coordinator_protect_top_k]
                        }

                    coordinator_disabled_for_query = False
                    margin_top_k = self.coordinator_margin_gate_top_k
                    if self.use_learned_coordinator and margin_top_k > 0 and len(candidate_records) > margin_top_k:
                        ranked_heuristics = sorted(
                            (record[2] for record in candidate_records),
                            reverse=True,
                        )
                        boundary_margin = ranked_heuristics[margin_top_k - 1] - ranked_heuristics[margin_top_k]
                        safe_threshold = self.coordinator_margin_gate_threshold
                        if safe_threshold <= 0:
                            safe_threshold = self.coordinator_blend_alpha
                        self.repair_stats["coordinator_margin_gate_checked"] += 1
                        if boundary_margin <= safe_threshold:
                            coordinator_disabled_for_query = True
                            self.repair_stats["coordinator_margin_gate_disabled"] += 1

                    final_scores = {}
                    coordinator_active = (
                        self.rerank_fusion_mode != "query_conditioned"
                        and
                        self.use_learned_coordinator
                        and (
                            self.coordinator_weights is not None
                            or self.coordinator_mlp_state is not None
                        )
                        and self.coordinator_feature_mean is not None
                        and self.coordinator_feature_std is not None
                    )
                    for passage, info, heuristic_score in candidate_records:
                        if (
                                not coordinator_active
                                or coordinator_disabled_for_query
                                or passage in protected_passages):
                            final_scores[passage] = heuristic_score
                        else:
                            final_scores[passage] = self._score_candidate_with_coordinator(
                                passage=passage,
                                info=info,
                                query_emb_tensor=query_emb_tensor,
                                ppr_scores_dict=ppr_scores_dict,
                                heuristic_score=heuristic_score,
                            )

                    if not final_scores:
                        sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                    else:
                        final_sorted_passages = sorted(
                            final_scores,
                            key=lambda passage: (final_scores[passage], passage),
                            reverse=True,
                        )
                        if (
                                self.rerank_fusion_mode == "query_conditioned"
                                and self.set_aware_selector is not None):
                            final_sorted_passages = self._apply_set_aware_selection(
                                candidate_components,
                                final_scores,
                                query,
                            )
                        protected_head = []
                        if (
                                self.rerank_fusion_mode != "query_conditioned"
                                and self.similarity_head_protect_k > 0):
                            similarity_head_scores = {
                                record["passage"]: record["similarity"]
                                for record in candidate_components
                                if record["similarity"] is not None
                            }
                            protected_head.extend(sorted(
                                similarity_head_scores,
                                key=lambda passage: (
                                    similarity_head_scores[passage],
                                    passage,
                                ),
                                reverse=True,
                            )[:self.similarity_head_protect_k])
                        if (
                                self.rerank_fusion_mode != "query_conditioned"
                                and self.fact_head_protect_k > 0):
                            fact_head_scores = {
                                record["passage"]: (
                                    self.w_policy * record["fact"]
                                    + self.w_sim * record["similarity"]
                                )
                                for record in candidate_components
                                if record["fact"] is not None
                            }
                            protected_fact_head = sorted(
                                fact_head_scores,
                                key=lambda passage: (
                                    fact_head_scores[passage],
                                    passage,
                                ),
                                reverse=True,
                            )[:self.fact_head_protect_k]
                            protected_head.extend(
                                passage for passage in protected_fact_head
                                if passage not in protected_head
                            )
                        if protected_head:
                            protected_set = set(protected_head)
                            final_sorted_passages = protected_head + [
                                passage for passage in final_sorted_passages
                                if passage not in protected_set
                            ]
                        passage_key_to_idx = {key: i for i, key in enumerate(self.passage_node_keys)}
                        sorted_doc_ids = np.array([passage_key_to_idx[p] for p in final_sorted_passages if p in passage_key_to_idx])
                        sorted_doc_scores = np.array([final_scores[p] for p in final_sorted_passages if p in passage_key_to_idx])
    
                # Retrieve actual content based on sorted IDs
                top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
                retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))
                if self._repair_triggered_current_query:
                    triggered_query_indices.append(q_idx)
                if (
                    self._repair_gate_eligible_current
                    and self._repair_triggered_current_query
                    and self._repair_gate_features_current is not None
                    and gold_docs is not None
                ):
                    self.repair_gate_records.append({
                        "query": query,
                        "features": self._repair_gate_features_current,
                        "label": int(novel_gold_repair_count > 0),
                        "novel_gold_repair_passages": novel_gold_repair_count,
                        "repair_passages": len(repair_info),
                        "gate_probability": self._repair_gate_probability_current,
                    })
    
            all_retrieval_time = time.time() - retrieve_start_time
            
            if gold_docs is not None:
                k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
                retrieved_docs = [r.docs for r in retrieval_results]
                overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                    gold_docs=gold_docs,
                    retrieved_docs=retrieved_docs,
                    k_list=k_list,
                )
                triggered_subset_result = {}
                if triggered_query_indices:
                    for k in k_list:
                        metric_name = f"Recall@{k}"
                        triggered_subset_result[metric_name] = round(float(np.mean([
                            example_retrieval_results[idx][metric_name]
                            for idx in triggered_query_indices
                        ])), 4)
                self.repair_stats["triggered_query_count"] = len(triggered_query_indices)
                self.last_repair_triggered_query_indices = list(triggered_query_indices)
                self.last_source_diagnostics = {}
                if source_diagnostic_docs is not None:
                    for source_name, source_docs in source_diagnostic_docs.items():
                        source_metrics, _ = retrieval_recall_evaluator.calculate_metric_scores(
                            gold_docs=gold_docs,
                            retrieved_docs=source_docs,
                            k_list=k_list,
                        )
                        complete_evidence = self._complete_evidence_diagnostics(
                            gold_docs,
                            source_docs,
                            k_list,
                        )
                        self.last_source_diagnostics[source_name] = {
                            "recall": source_metrics,
                            "complete_evidence": complete_evidence,
                        }
                self._save_repair_gate_records()
                
                logger.info(f"Total Retrieval Time: {all_retrieval_time:.2f}s")
                logger.info(f"CoEviRAG repair stats: {self.repair_stats}")
                logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
                logger.info(f"Repair-triggered subset retrieval results: {triggered_subset_result}")
                logger.info(
                    "Paragraph search stats: %s",
                    self.last_paragraph_search_stats,
                )
                logger.info(
                    "Fact traversal stats: %s",
                    self.fact_traversal_stats,
                )
                if getattr(self.global_config, "rl_paragraph_seed_gate_enabled", False):
                    gate_stats = dict(self.last_fact_seed_gate_stats)
                    checked = max(1, int(gate_stats.get("checked", 0)))
                    gate_stats["mean_keep_probability"] = (
                        float(gate_stats.pop("keep_probability_sum", 0.0)) / checked
                    )
                    logger.info("Fact-to-Paragraph seed gate stats: %s", gate_stats)
                if self.last_source_diagnostics:
                    logger.info(
                        "Source retrieval diagnostics: %s",
                        self.last_source_diagnostics,
                    )
                return retrieval_results, overall_retrieval_result
                
            return retrieval_results

    @staticmethod
    def _complete_evidence_diagnostics(
            gold_docs: List[List[str]],
            retrieved_docs: List[List[str]],
            k_list: List[int]) -> Dict[str, Any]:
        eligible = sum(1 for gold in gold_docs if len(set(gold)) >= 2)
        complete_at_k = {}
        for k in k_list:
            complete = 0
            for gold, retrieved in zip(gold_docs, retrieved_docs):
                gold_set = set(gold)
                if len(gold_set) < 2:
                    continue
                if gold_set.issubset(set(retrieved[:k])):
                    complete += 1
            complete_at_k[f"CompleteEvidence@{k}"] = round(
                complete / eligible if eligible else 0.0,
                4,
            )

        second_ranks = []
        for gold, retrieved in zip(gold_docs, retrieved_docs):
            gold_set = set(gold)
            if len(gold_set) < 2:
                continue
            rank_by_doc = {}
            for rank, doc in enumerate(retrieved, start=1):
                rank_by_doc.setdefault(doc, rank)
            if gold_set.issubset(rank_by_doc):
                second_ranks.append(max(rank_by_doc[doc] for doc in gold_set))
        return {
            "eligible_queries": eligible,
            "all_gold_found": len(second_ranks),
            "second_gold_rank_mean": round(float(np.mean(second_ranks)), 4)
            if second_ranks else None,
            "second_gold_rank_median": round(float(np.median(second_ranks)), 4)
            if second_ranks else None,
            **complete_at_k,
        }


    def rag_qa(self, queries: List[Union[str, QuerySolution]], gold_docs: List[List[str]] = None, gold_answers: List[List[str]] = None):
        """
        Executes the full RAG pipeline: Retrieval followed by Question Answering, including evaluation against gold answers.
        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)
        
        overall_retrieval_result = {}
        if not isinstance(queries[0], QuerySolution):
            retrieval_output = self.retrieve(queries=queries, gold_docs=gold_docs)
            if gold_docs is not None:
                queries, overall_retrieval_result = retrieval_output
            else:
                queries = retrieval_output
        
        queries_solutions, all_response_message, all_metadata = self.qa(queries)
        
        if gold_answers is not None:
            overall_qa_em_result, _ = qa_em_evaluator.calculate_metric_scores(gold_answers=gold_answers, predicted_answers=[q.answer for q in queries_solutions], aggregation_fn=np.max)
            overall_qa_f1_result, _ = qa_f1_evaluator.calculate_metric_scores(gold_answers=gold_answers, predicted_answers=[q.answer for q in queries_solutions], aggregation_fn=np.max)
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_em_result.items()}
       
            logger.info(f"Evaluation results for QA: {overall_qa_results}")
            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        
        return queries_solutions, all_response_message, all_metadata

    def get_supporting_passages_for_node(
            self, node_key: str, include_masked: bool = False) -> Set[str]:
        """Return paragraph ids supporting a fact/entity node."""
        cache = self.node_supporting_passages_cache
        if node_key not in cache:
            supporting_passages = set()
            if node_key in getattr(self, "passage_key_to_idx", {}):
                supporting_passages.add(node_key)
            supporting_passages.update(
                getattr(self, "ent_node_to_chunk_ids", {}).get(node_key, set())
            )
            supporting_passages.update(
                getattr(self, "fact_node_to_chunk_ids", {}).get(node_key, set())
            )
            cache[node_key] = frozenset(supporting_passages)

        supporting_passages = cache[node_key]
        masked_passages = getattr(self, "base_retrieval_masked_passages", set())
        if include_masked or not masked_passages:
            return supporting_passages
        return supporting_passages.difference(masked_passages)

    @staticmethod
    def _new_gold_hop_diagnostics() -> Dict[str, Any]:
        return {
            "eligible_triggered_queries": 0,
            "triggered_without_missing_gold": 0,
            "no_accepted_overlay": 0,
            "endpoint_link_failure": 0,
            "missing_gold_without_kg_nodes": 0,
            "full_distance": {"0": 0, "1": 0, "2": 0, "3": 0, "unreachable": 0},
            "topk_distance": {"0": 0, "1": 0, "2": 0, "3": 0, "unreachable": 0},
            "examples": [],
        }

    def _passage_keys_for_contents(self, contents: List[str]) -> Set[str]:
        content_set = set(contents or [])
        if not content_set:
            return set()
        return {
            key for key, row in self.chunk_embedding_store.hash_id_to_row.items()
            if row.get("content") in content_set
        }

    def _prepare_gold_hop_diagnostic(
            self, gold_contents: Optional[List[str]], base_passages: Set[str]) -> None:
        self._repair_gold_hop_current = None
        if not getattr(self.global_config, "repair_gold_hop_diagnostics", False) or not gold_contents:
            return
        gold_passages = self._passage_keys_for_contents(gold_contents)
        self._repair_gold_hop_current = {
            "missing_gold": gold_passages - set(base_passages),
            "expansion_seeds": set(),
            "accepted_overlay": 0,
            "generated_candidates": 0,
            "verification_rejected": 0,
            "unanchored_candidates": 0,
        }

    def _gold_support_nodes(self, passage_keys: Set[str]) -> Set[str]:
        support_nodes = set()
        for passage_key in passage_keys:
            support_nodes.update(getattr(self, "chunk_to_entity_node_ids", {}).get(passage_key, set()))
            support_nodes.update(getattr(self, "chunk_to_fact_node_ids", {}).get(passage_key, set()))
        graph_nodes = getattr(self, "node_name_to_vertex_idx", {})
        return {key for key in support_nodes if key in graph_nodes}

    def _bounded_gold_distance(
            self,
            seed_nodes: Set[str],
            target_nodes: Set[str],
            max_hops: int,
            query_emb_tensor: Optional[torch.Tensor] = None,
            branch_top_k: Optional[int] = None) -> Optional[int]:
        graph_nodes = getattr(self, "node_name_to_vertex_idx", {})
        frontier = {key for key in seed_nodes if key in graph_nodes}
        visited = set()
        for distance in range(max_hops + 1):
            frontier.difference_update(visited)
            if not frontier:
                return None
            if frontier.intersection(target_nodes):
                return distance
            visited.update(frontier)
            next_frontier = set()
            for node_key in frontier:
                if branch_top_k is None:
                    neighbors = list(self._get_fact_neighbors(node_key))
                else:
                    neighbors = self._rank_overlay_neighbors(
                        node_key,
                        query_emb_tensor=query_emb_tensor,
                        top_k=branch_top_k,
                    )
                next_frontier.update(neighbors)
            frontier = next_frontier
        return None

    def _finalize_gold_hop_diagnostic(self, query: str) -> None:
        current = self._repair_gold_hop_current
        self._repair_gold_hop_current = None
        if current is None:
            return
        stats = self.repair_stats["gold_hop_diagnostics"]
        missing_gold = current["missing_gold"]
        if not missing_gold:
            stats["triggered_without_missing_gold"] += 1
            return
        stats["eligible_triggered_queries"] += 1
        example = {
            "query": query,
            "missing_gold_count": len(missing_gold),
        }
        if current["accepted_overlay"] == 0:
            stats["no_accepted_overlay"] += 1
            stats["full_distance"]["unreachable"] += 1
            stats["topk_distance"]["unreachable"] += 1
            example.update({
                "status": "no_accepted_overlay",
                "generated_candidates": current["generated_candidates"],
                "verification_rejected": current["verification_rejected"],
                "unanchored_candidates": current["unanchored_candidates"],
            })
            if len(stats["examples"]) < 5:
                stats["examples"].append(example)
            return
        seeds = current["expansion_seeds"]
        if not seeds:
            stats["endpoint_link_failure"] += 1
            stats["full_distance"]["unreachable"] += 1
            stats["topk_distance"]["unreachable"] += 1
            example["status"] = "endpoint_link_failure"
            if len(stats["examples"]) < 5:
                stats["examples"].append(example)
            return
        target_nodes = self._gold_support_nodes(missing_gold)
        if not target_nodes:
            stats["missing_gold_without_kg_nodes"] += 1
            stats["full_distance"]["unreachable"] += 1
            stats["topk_distance"]["unreachable"] += 1
            example["status"] = "missing_gold_without_kg_nodes"
            if len(stats["examples"]) < 5:
                stats["examples"].append(example)
            return

        max_hops = max(0, int(getattr(self.global_config, "repair_gold_hop_max_hops", 3)))
        full_distance = self._bounded_gold_distance(seeds, target_nodes, max_hops)
        query_emb_tensor = self._get_query_embedding_tensor(query)
        topk_distance = self._bounded_gold_distance(
            seeds,
            target_nodes,
            max_hops,
            query_emb_tensor=query_emb_tensor,
            branch_top_k=max(1, int(getattr(self.global_config, "repair_overlay_branch_top_k", 8))),
        )
        stats["full_distance"][str(full_distance) if full_distance is not None else "unreachable"] += 1
        stats["topk_distance"][str(topk_distance) if topk_distance is not None else "unreachable"] += 1
        example.update({
            "status": "measured",
            "full_distance": full_distance,
            "topk_distance": topk_distance,
        })
        if len(stats["examples"]) < 5:
            stats["examples"].append(example)

    def _remove_fact_nodes_in_memory(self, fact_keys: Set[str]) -> Set[str]:
        """Remove fact vertices and keep all retrieval-side indexes aligned."""
        graph_nodes = getattr(self, "node_name_to_vertex_idx", {})
        selected_facts = {key for key in fact_keys if key in graph_nodes}
        if not selected_facts:
            return set()

        vertex_ids = sorted(
            [graph_nodes[key] for key in selected_facts],
            reverse=True,
        )
        old_fact_keys = list(getattr(self, "fact_node_keys", []))
        keep_mask = np.array(
            [key not in selected_facts for key in old_fact_keys],
            dtype=bool,
        )
        self.fact_node_keys = [
            key for key in old_fact_keys if key not in selected_facts
        ]
        if len(getattr(self, "fact_embeddings", [])) == len(old_fact_keys):
            self.fact_embeddings = self.fact_embeddings[keep_mask]

        self.graph.delete_vertices(vertex_ids)
        self.node_name_to_vertex_idx = {
            vertex["name"]: vertex.index for vertex in self.graph.vs
        }
        self.entity_node_idxs = [
            self.node_name_to_vertex_idx[key]
            for key in getattr(self, "entity_node_keys", [])
            if key in self.node_name_to_vertex_idx
        ]
        self.fact_node_idxs = [
            self.node_name_to_vertex_idx[key]
            for key in self.fact_node_keys
            if key in self.node_name_to_vertex_idx
        ]

        for fact_key in selected_facts:
            self.fact_node_to_chunk_ids.pop(fact_key, None)
            self.fact_node_to_triple.pop(fact_key, None)
        for passage_key, passage_fact_keys in getattr(
            self, "chunk_to_fact_node_ids", {}
        ).items():
            self.chunk_to_fact_node_ids[passage_key] = (
                set(passage_fact_keys) - selected_facts
            )

        self._repair_endpoint_fact_index = None
        self.paragraph_neighbor_cache.clear()
        self.fact_neighbor_cache.clear()
        self.fact_score_cache.clear()
        self.fact_endpoint_cache.clear()
        self.fact_rank_cache.clear()
        self.node_supporting_passages_cache.clear()
        self.node_structural_score_cache.clear()
        return selected_facts

    def apply_fact_node_dropout(
            self,
            rate: float = 0.0,
            seed: int = 42) -> Dict[str, Any]:
        """Remove a deterministic nested subset of complete OpenIE fact nodes."""
        rate = float(np.clip(rate, 0.0, 1.0))
        candidate_facts = list(getattr(self, "fact_node_keys", []))
        selected_facts = set()
        for fact_key in candidate_facts:
            digest = hashlib.md5(
                f"{seed}|fact-node|{fact_key}".encode("utf-8")
            ).hexdigest()
            unit_value = int(digest[:16], 16) / float(16 ** 16)
            if unit_value < rate:
                selected_facts.add(fact_key)

        removed_facts = self._remove_fact_nodes_in_memory(selected_facts)
        stats = {
            "rate": rate,
            "seed": int(seed),
            "candidate_fact_nodes": len(candidate_facts),
            "removed_fact_nodes": len(removed_facts),
            "actual_rate": (
                len(removed_facts) / len(candidate_facts)
                if candidate_facts else 0.0
            ),
            "remaining_fact_nodes": len(getattr(self, "fact_node_keys", [])),
            "remaining_graph_edges": self.graph.ecount(),
        }
        self.fact_node_dropout_stats = stats
        logger.info(f"Applied in-memory fact-node dropout: {stats}")
        return stats

    def apply_fact_edge_dropout(self, rate: float = 0.0, seed: int = 42) -> Dict[str, Any]:
        """Drop a deterministic nested subset of entity-fact edges in memory only."""
        rate = float(np.clip(rate, 0.0, 1.0))
        fact_keys = set(getattr(self, "fact_node_keys", []))
        candidate_edges = []
        selected_edge_ids = []

        for edge in self.graph.es:
            source_idx, target_idx = edge.tuple
            source_name = self.graph.vs[source_idx]["name"]
            target_name = self.graph.vs[target_idx]["name"]
            source_is_fact = source_name in fact_keys
            target_is_fact = target_name in fact_keys
            if source_is_fact == target_is_fact:
                continue

            candidate_edges.append(edge.index)
            edge_key = "|".join(sorted([source_name, target_name]))
            digest = hashlib.md5(f"{seed}|{edge_key}".encode("utf-8")).hexdigest()
            unit_value = int(digest[:16], 16) / float(16 ** 16)
            if unit_value < rate:
                selected_edge_ids.append(edge.index)

        if selected_edge_ids:
            self.graph.delete_edges(selected_edge_ids)
            self._repair_endpoint_fact_index = None
            self.fact_neighbor_cache.clear()
            self.node_structural_score_cache.clear()

        stats = {
            "rate": rate,
            "seed": int(seed),
            "candidate_fact_edges": len(candidate_edges),
            "dropped_fact_edges": len(selected_edge_ids),
            "actual_rate": (
                len(selected_edge_ids) / len(candidate_edges) if candidate_edges else 0.0
            ),
            "remaining_graph_edges": self.graph.ecount(),
        }
        self.fact_edge_dropout_stats = stats
        logger.info(f"Applied in-memory fact-edge dropout: {stats}")
        return stats

    def apply_targeted_bridge_fact_dropout(
            self,
            queries: List[str],
            gold_docs: List[List[str]],
            facts_per_query: int = 1,
            seed: int = 42,
            mask_supporting_passages: bool = False) -> Dict[str, Any]:
        """Remove gold-document bridge facts in memory while retaining source paragraphs."""
        facts_per_query = max(1, int(facts_per_query))
        content_to_passage = {}
        for passage_key in getattr(self, "passage_node_keys", []):
            try:
                content = self.chunk_embedding_store.get_row(passage_key)["content"]
            except Exception:
                continue
            content_to_passage[content] = passage_key

        selected_facts = set()
        masked_passages = set()
        targeted_queries = 0
        bridge_targeted_queries = 0
        fallback_targeted_queries = 0
        required_slot_targeted_queries = 0

        def deterministic_order(query: str, fact_key: str) -> str:
            return hashlib.md5(f"{seed}|{query}|{fact_key}".encode("utf-8")).hexdigest()

        for query, query_gold_docs in zip(queries, gold_docs):
            gold_passages = [
                content_to_passage[content]
                for content in (query_gold_docs or [])
                if content in content_to_passage
            ]
            if not gold_passages:
                continue

            entity_counts = defaultdict(int)
            for passage_key in gold_passages:
                for entity_key in getattr(self, "chunk_to_entity_node_ids", {}).get(passage_key, set()):
                    entity_counts[entity_key] += 1
            shared_entities = {key for key, count in entity_counts.items() if count >= 2}

            bridge_candidates = set()
            all_gold_facts = set()
            for passage_key in gold_passages:
                for fact_key in getattr(self, "chunk_to_fact_node_ids", {}).get(passage_key, set()):
                    all_gold_facts.add(fact_key)
                    triple = getattr(self, "fact_node_to_triple", {}).get(fact_key)
                    if not triple or len(triple) != 3:
                        continue
                    endpoint_keys = {
                        self._overlay_entity_key(triple[0]),
                        self._overlay_entity_key(triple[2]),
                    }
                    if endpoint_keys.intersection(shared_entities):
                        bridge_candidates.add(fact_key)

            candidates = bridge_candidates or all_gold_facts
            if not candidates:
                continue
            required_families = {
                self._repair_relation_family(triple[1], query=query)
                for triple in self._get_raw_query_triples(query)
            }

            def target_priority(fact_key: str) -> Tuple[int, str]:
                triple = getattr(self, "fact_node_to_triple", {}).get(fact_key)
                if triple and len(triple) == 3:
                    family = self._repair_relation_family(triple[1], query=query, tail=triple[2])
                    if family in required_families:
                        return 0, deterministic_order(query, fact_key)
                return 1, deterministic_order(query, fact_key)

            ordered = sorted(candidates, key=target_priority)
            chosen = ordered[:facts_per_query]
            selected_facts.update(chosen)
            if mask_supporting_passages:
                gold_passage_set = set(gold_passages)
                for fact_key in chosen:
                    masked_passages.update(
                        set(getattr(self, "fact_node_to_chunk_ids", {}).get(fact_key, set()))
                        .intersection(gold_passage_set)
                    )
            targeted_queries += 1
            if chosen and target_priority(chosen[0])[0] == 0:
                required_slot_targeted_queries += 1
            if bridge_candidates:
                bridge_targeted_queries += 1
            else:
                fallback_targeted_queries += 1

        selected_facts = self._remove_fact_nodes_in_memory(selected_facts)
        self.base_retrieval_masked_passages = set(masked_passages)

        stats = {
            "seed": int(seed),
            "facts_per_query": facts_per_query,
            "targeted_queries": targeted_queries,
            "bridge_targeted_queries": bridge_targeted_queries,
            "fallback_targeted_queries": fallback_targeted_queries,
            "required_slot_targeted_queries": required_slot_targeted_queries,
            "removed_unique_fact_nodes": len(selected_facts),
            "masked_base_passages": len(self.base_retrieval_masked_passages),
            "remaining_fact_nodes": len(self.fact_node_keys),
            "remaining_graph_edges": self.graph.ecount(),
        }
        self.targeted_bridge_fact_dropout_stats = stats
        logger.info(f"Applied targeted bridge-fact dropout: {stats}")
        return stats

    def apply_query_relation_fact_dropout(
            self,
            queries: List[str],
            facts_per_query: int = 1,
            seed: int = 42,
            mask_source_passages: bool = False) -> Dict[str, Any]:
        """Remove query-matched relation facts without consulting gold evidence."""
        facts_per_query = max(1, int(facts_per_query))
        graph_nodes = getattr(self, "node_name_to_vertex_idx", {})
        endpoint_index = defaultdict(list)
        for fact_key, raw_triple in getattr(self, "fact_node_to_triple", {}).items():
            if len(raw_triple) != 3 or fact_key not in graph_nodes:
                continue
            if self.graph.degree(graph_nodes[fact_key]) < 2:
                continue
            triple = tuple(str(value) for value in raw_triple)
            family = self._repair_relation_family(triple[1], tail=triple[2])
            for endpoint in (triple[0], triple[2]):
                for alias in self._repair_entity_aliases(endpoint):
                    endpoint_index[alias].append((fact_key, triple, family))

        selected_facts = set()
        masked_passages = set()
        targeted_queries = 0
        queries_with_extracted_slots = 0
        candidate_facts = 0
        relation_family_counts = defaultdict(int)

        def deterministic_order(query: str, fact_key: str) -> str:
            return hashlib.md5(
                f"{seed}|query-relation|{query}|{fact_key}".encode("utf-8")
            ).hexdigest()

        for query in queries:
            slots = []
            for raw_triple in self._get_raw_query_triples(query):
                head = self._repair_normalize_entity(raw_triple[0])
                relation = str(raw_triple[1])
                if not head:
                    continue
                family = self._repair_relation_family(relation, query=query)
                if family == "comparison_operator":
                    continue
                slots.append((head, relation, family))
            if not slots:
                continue
            queries_with_extracted_slots += 1

            query_candidates = {}
            for slot_index, (head, relation, family) in enumerate(slots):
                relation_tokens = set(self._repair_normalize_entity(relation).split())
                for alias in self._repair_entity_aliases(head):
                    for fact_key, triple, fact_family in endpoint_index.get(alias, []):
                        if fact_key in selected_facts or fact_family != family:
                            continue
                        fact_relation_tokens = set(
                            self._repair_normalize_entity(triple[1]).split()
                        )
                        overlap = len(relation_tokens.intersection(fact_relation_tokens))
                        priority = (
                            slot_index,
                            -overlap,
                            deterministic_order(query, fact_key),
                        )
                        previous = query_candidates.get(fact_key)
                        if previous is None or priority < previous[0]:
                            query_candidates[fact_key] = (priority, family)

            candidate_facts += len(query_candidates)
            ordered = sorted(
                query_candidates,
                key=lambda fact_key: query_candidates[fact_key][0],
            )
            chosen = ordered[:facts_per_query]
            if not chosen:
                continue
            targeted_queries += 1
            selected_facts.update(chosen)
            for fact_key in chosen:
                relation_family_counts[query_candidates[fact_key][1]] += 1
                if mask_source_passages:
                    masked_passages.update(
                        getattr(self, "fact_node_to_chunk_ids", {}).get(fact_key, set())
                    )

        selected_facts = self._remove_fact_nodes_in_memory(selected_facts)
        self.base_retrieval_masked_passages = set(masked_passages)
        stats = {
            "seed": int(seed),
            "facts_per_query": facts_per_query,
            "queries": len(queries),
            "queries_with_extracted_slots": queries_with_extracted_slots,
            "targeted_queries": targeted_queries,
            "candidate_fact_assignments": candidate_facts,
            "removed_unique_fact_nodes": len(selected_facts),
            "masked_source_passages": len(self.base_retrieval_masked_passages),
            "relation_family_counts": dict(sorted(relation_family_counts.items())),
            "remaining_fact_nodes": len(self.fact_node_keys),
            "remaining_graph_edges": self.graph.ecount(),
            "uses_gold_annotations": False,
        }
        self.query_relation_fact_dropout_stats = stats
        logger.info(f"Applied query-relation fact dropout: {stats}")
        return stats

    def _node_structural_score(self, node_key: str) -> float:
        if node_key in self.node_structural_score_cache:
            return self.node_structural_score_cache[node_key]
        if node_key not in getattr(self, "node_name_to_vertex_idx", {}):
            return 0.0
        try:
            degree = self.graph.degree(self.node_name_to_vertex_idx[node_key])
            score = float(np.log1p(degree))
            self.node_structural_score_cache[node_key] = score
            return score
        except Exception:
            return 0.0

    def _path_evidence_state(self, path: List[str], path_score: float) -> Dict:
        evidence_nodes, passages, structural_score = [], set(), 0.0
        for node_key in path:
            node_passages = self.get_supporting_passages_for_node(node_key)
            if node_passages:
                evidence_nodes.append(node_key)
                passages.update(node_passages)
            structural_score += self._node_structural_score(node_key)
        return {
            "path": tuple(path), "evidence_nodes": evidence_nodes,
            "passages": passages, "structural_score": structural_score,
            "score": path_score + min(len(evidence_nodes), 6) * 0.05
                     + min(structural_score, 12.0) * 0.01,
        }

    def _update_passage_info_from_path(
            self,
            passage_info: Dict[str, Dict],
            path: List[str],
            path_score: float,
            source: str = "fact_path",
            confidence: float = 0.0,
            prefix_evidence: Optional[Dict] = None) -> None:
        """Map a fact/entity path to paragraph candidates via side indexes."""
        evidence_nodes = []
        candidate_passages = set()
        structural_score = 0.0

        if prefix_evidence is not None:
            if tuple(path[:-1]) != prefix_evidence["path"] or source != "fact_path" or confidence != 0.0:
                raise ValueError("Prefix reuse requires an already mapped Fact path")
            evidence_nodes = list(prefix_evidence["evidence_nodes"])
            structural_score = prefix_evidence["structural_score"]
            nodes_to_map = path[-1:]
        else:
            nodes_to_map = path

        for node_key in nodes_to_map:
            node_passages = self.get_supporting_passages_for_node(node_key)
            if node_passages:
                evidence_nodes.append(node_key)
                candidate_passages.update(node_passages)
            structural_score += self._node_structural_score(node_key)

        if not candidate_passages and prefix_evidence is None:
            return

        coverage_bonus = min(len(evidence_nodes), 6) * 0.05
        structural_bonus = min(structural_score, 12.0) * 0.01
        repair_bonus = confidence * 0.25 if source == "repair" else 0.0
        candidate_score = path_score + coverage_bonus + structural_bonus + repair_bonus

        if prefix_evidence is not None:
            # Prefix passages already have at least the prefix score; equal scores never replace paths.
            if candidate_score <= prefix_evidence["score"]:
                candidate_passages.difference_update(prefix_evidence["passages"])
            else:
                candidate_passages.update(prefix_evidence["passages"])

        for passage_key in sorted(candidate_passages):
            current = passage_info.get(passage_key)
            if current is None or candidate_score > current.get("score", -float("inf")):
                passage_info[passage_key] = {
                    "score": candidate_score,
                    "path": list(path),
                    "source": source,
                    "agent": "fact",
                    "confidence": confidence,
                    "evidence_nodes": list(evidence_nodes),
                    "ppr_score": structural_bonus,
                }

    def _map_fact_path_extensions(
            self, passage_info: Dict[str, Dict], extensions: List[Dict],
            prefix_evidence: Dict) -> None:
        """Map ordered siblings without rebuilding their common prefix per child."""
        prefix_passages = prefix_evidence["passages"]
        prefix_nodes = prefix_evidence["evidence_nodes"]
        prefix_structural = prefix_evidence["structural_score"]
        prefix_score = prefix_evidence["score"]
        support = self.get_supporting_passages_for_node
        structural = self._node_structural_score
        current_info = passage_info.get
        for extension in extensions:
            path = extension["path"]
            node = path[-1]
            node_passages = support(node)
            evidence_count = len(prefix_nodes) + bool(node_passages)
            structural_bonus = min(prefix_structural + structural(node), 12.0) * 0.01
            candidate_score = extension["score"] + min(evidence_count, 6) * 0.05 + structural_bonus
            if candidate_score <= prefix_score:
                candidates = (key for key in sorted(node_passages) if key not in prefix_passages)
            else:
                candidates = sorted(prefix_passages.union(node_passages))
            evidence_nodes = None
            for passage_key in candidates:
                current = current_info(passage_key)
                if current is None or candidate_score > current.get("score", -float("inf")):
                    if evidence_nodes is None:
                        evidence_nodes = list(prefix_nodes)
                        if node_passages:
                            evidence_nodes.append(node)
                    passage_info[passage_key] = {
                        "score": candidate_score,
                        "path": list(path),
                        "source": "fact_path",
                        "agent": "fact",
                        "confidence": 0.0,
                        "evidence_nodes": list(evidence_nodes),
                        "ppr_score": structural_bonus,
                    }

    def _evidence_semantic_neighbors(self, passage_idx: int, top_k: int) -> Tuple[int, ...]:
        """Cache only immutable KNN indices, never query scores or masked results."""
        optimized = getattr(self, "optimize_static_graph_ops", False)
        if optimized:
            signature = (id(self.passage_embeddings), self.passage_embeddings.shape,
                         id(self.passage_node_keys), len(self.passage_node_keys))
            if signature != self.evidence_semantic_cache_signature:
                self.evidence_semantic_neighbor_cache.clear()
                self.evidence_semantic_cache_signature = signature
            key = (passage_idx, top_k)
            cached = self.evidence_semantic_neighbor_cache.get(key)
            if cached is not None:
                return cached
        scores = np.dot(self.passage_embeddings, self.passage_embeddings[passage_idx].T)
        scores = np.squeeze(scores) if scores.ndim == 2 else scores
        indices = tuple(int(idx) for idx in np.argsort(scores)[-(top_k + 1):][::-1])
        if optimized:
            self.evidence_semantic_neighbor_cache[key] = indices
        return indices

    def _search_scores_to_cpu(self, groups: List[torch.Tensor]) -> List[List[float]]:
        if not groups:
            return []
        if not getattr(self, "optimize_static_graph_ops", False):
            return [group.detach().cpu().tolist() for group in groups]
        lengths = [group.numel() for group in groups]
        values = torch.cat([group.detach().reshape(-1) for group in groups]).cpu().tolist()
        result, offset = [], 0
        for length in lengths:
            result.append(values[offset:offset + length])
            offset += length
        return result

    def _finalize_fact_passages(
            self, passage_info: Dict[str, Dict], query_emb_tensor: torch.Tensor,
            include_masked: bool = False) -> Dict[str, Dict]:
        if not getattr(self.global_config, "fact_return_paragraph_expansion", True):
            return passage_info
        return self._expand_evidence_paragraphs(
            passage_info, query_emb_tensor, include_masked=include_masked,
        )

    def _expand_evidence_paragraphs(
            self,
            passage_info: Dict[str, Dict],
            query_emb_tensor: torch.Tensor,
            max_seed_passages: int = 50,
            include_masked: bool = False) -> Dict[str, Dict]:
        """Lightweight paragraph-only evidence graph expansion."""
        paragraph_topk = getattr(self.global_config, "paragraph_graph_topk", 2)
        if paragraph_topk <= 0 or not passage_info or len(self.passage_embeddings) == 0:
            return passage_info

        passage_key_to_idx = (
            self.passage_key_to_idx if getattr(self, "optimize_static_graph_ops", False)
            else {key: i for i, key in enumerate(self.passage_node_keys)}
        )
        seed_items = sorted(
            passage_info.items(),
            key=lambda item: (item[1].get("score", 0.0), item[0]),
            reverse=True,
        )[:max_seed_passages]
        query_vec = query_emb_tensor.detach().cpu().numpy().reshape(-1)

        for passage_key, info in seed_items:
            if (
                    not include_masked
                    and passage_key in getattr(self, "base_retrieval_masked_passages", set())):
                continue
            passage_idx = passage_key_to_idx.get(passage_key)
            if passage_idx is None:
                continue
            top_neighbor_indices = self._evidence_semantic_neighbors(passage_idx, paragraph_topk)

            for neighbor_idx in top_neighbor_indices:
                neighbor_key = self.passage_node_keys[int(neighbor_idx)]
                if neighbor_key == passage_key:
                    continue
                if (
                        not include_masked
                        and neighbor_key in getattr(self, "base_retrieval_masked_passages", set())):
                    continue
                query_sim = float(np.dot(self.passage_embeddings[int(neighbor_idx)], query_vec.T))
                expanded_score = info.get("score", 0.0) * 0.85 + query_sim * 0.15
                current = passage_info.get(neighbor_key)
                if current is None or expanded_score > current.get("score", -float("inf")):
                    passage_info[neighbor_key] = {
                        "score": expanded_score,
                        "path": info.get("path", []),
                        "source": "paragraph_graph",
                        "confidence": info.get("confidence", 0.0),
                        "evidence_nodes": info.get("evidence_nodes", []),
                        "ppr_score": info.get("ppr_score", 0.0),
                    }

        return passage_info





    def _reset_repair_stats(self) -> None:
        self.repair_stats = {
            "triggered": 0,
            "accepted": 0,
            "rejected": 0,
            "virtual_edges": 0,
            "relation_slots": 0,
            "resolved_slot_triples": 0,
            "unresolved_slots": 0,
            "verification_parse_failures": 0,
            "verification_retry_attempts": 0,
            "verification_retry_successes": 0,
            "trigger_examples": [],
            "structural_gate_checked": 0,
            "structural_gate_unparseable": 0,
            "required_slots": 0,
            "resolved_existing_slots": 0,
            "unresolved_required_slots": 0,
            "evidence_resolved_slots": 0,
            "retrieval_gap_slots": 0,
            "relation_evidence_matches": 0,
            "relation_evidence_misses": 0,
            "structural_gate_suppressed_queries": 0,
            "structural_gate_examples": [],
            "repair_passages": 0,
            "novel_repair_passages": 0,
            "gold_repair_passages": 0,
            "novel_gold_repair_passages": 0,
            "queries_with_novel_repair": 0,
            "queries_with_novel_gold_repair": 0,
            "overlay_anchored_triples": 0,
            "overlay_unanchored_triples": 0,
            "overlay_nodes": 0,
            "overlay_edges": 0,
            "overlay_linked_entities": 0,
            "overlay_expanded_nodes": 0,
            "overlay_expanded_passages": 0,
            "overlay_slot_guided_triples": 0,
            "semantic_expanded_passages": 0,
            "semantic_fallback_triples": 0,
            "overlay_unanchored_examples": [],
            "gold_hop_diagnostics": self._new_gold_hop_diagnostics(),
            "trigger_reasons": {},
            "coordinator_margin_gate_checked": 0,
            "coordinator_margin_gate_disabled": 0,
            "repair_gate_eligible": 0,
            "repair_gate_executed": 0,
            "repair_gate_skipped": 0,
        }
        self.repair_gate_records = []

    def _save_repair_gate_records(self):
        return None


















    def _repair_broken_path(self, *args, **kwargs):
        """Compatibility hook for exhausted beams; never calls an LLM."""
        return {}

    def _get_query_embedding_tensor(self, query: str) -> torch.Tensor:
        query_emb_obj = self.query_to_embedding['passage'].get(query, None)
        if query_emb_obj is None:
            query_emb_obj = self.embedding_model.batch_encode(
                query,
                instruction=get_query_instruction('query_to_passage'),
                norm=True
            )
            self.query_to_embedding['passage'][query] = query_emb_obj

        if isinstance(query_emb_obj, np.ndarray):
            query_emb = torch.from_numpy(query_emb_obj).float().to(self.device)
        else:
            query_emb = query_emb_obj.to(self.device)
        return query_emb.view(1, -1)


    def _prepare_paragraph_semantic_knn(self, top_k: int) -> bool:
        """Precompute semantic paragraph neighbors without materializing all pair scores."""
        if top_k in self.paragraph_semantic_knn_cache:
            return True
        if not self.precompute_paragraph_knn or self.device.type != "cuda":
            return False
        passage_count = len(getattr(self, "passage_node_keys", []))
        if passage_count == 0:
            return False

        embeddings = getattr(self, "passage_embeddings_tensor", None)
        owns_embeddings = embeddings is None
        if embeddings is None:
            embeddings = torch.from_numpy(self.passage_embeddings).float().to(self.device)

        candidate_count = min(top_k + 1, passage_count)
        neighbor_rows = [None] * passage_count
        started_at = time.time()
        logger.info(
            "Precomputing paragraph semantic Top-%d for %d passages (batch=%d).",
            candidate_count,
            passage_count,
            self.paragraph_knn_batch_size,
        )
        with torch.no_grad():
            key_embeddings = embeddings.transpose(0, 1)
            for start in range(0, passage_count, self.paragraph_knn_batch_size):
                end = min(start + self.paragraph_knn_batch_size, passage_count)
                similarities = torch.matmul(embeddings[start:end], key_embeddings)
                values, indices = torch.topk(
                    similarities,
                    k=candidate_count,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                values = values.cpu().numpy()
                indices = indices.cpu().numpy()
                for relative_idx in range(end - start):
                    neighbor_rows[start + relative_idx] = [
                        (int(idx), float(score))
                        for idx, score in zip(
                            indices[relative_idx], values[relative_idx]
                        )
                    ]
                del similarities, values, indices

        if owns_embeddings:
            del embeddings
        self.paragraph_semantic_knn_cache[top_k] = neighbor_rows
        logger.info(
            "Paragraph semantic Top-%d precomputed in %.2fs.",
            candidate_count,
            time.time() - started_at,
        )
        return True

    def _get_paragraph_neighbors(self, passage_key: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Return neighbors in the paragraph graph.

        The default graph is an implicit semantic KNN graph. When enabled, the
        evidence-aware variant augments semantic neighbors with paragraphs that
        share extracted facts or salient entities.
        """
        if top_k is None:
            top_k = getattr(self.global_config, "paragraph_graph_topk", 3)
        cache_key = (
            passage_key,
            top_k,
            bool(self.use_evidence_paragraph_graph),
            int(self.evidence_paragraph_max_candidates),
            int(self.evidence_paragraph_max_entity_support),
        )
        if cache_key in self.paragraph_neighbor_cache:
            return self.paragraph_neighbor_cache[cache_key]

        passage_idx = self.passage_key_to_idx.get(passage_key)
        if passage_idx is None or len(self.passage_embeddings) == 0:
            return []

        seed_vec = self.passage_embeddings[passage_idx]
        candidate_scores = {}

        def update_candidate(candidate_key: str, score: float) -> None:
            if candidate_key == passage_key:
                return
            if candidate_key in getattr(self, "base_retrieval_masked_passages", set()):
                return
            current = candidate_scores.get(candidate_key, -float("inf"))
            if score > current:
                candidate_scores[candidate_key] = float(score)

        scores = None
        if self._prepare_paragraph_semantic_knn(top_k):
            semantic_candidates = self.paragraph_semantic_knn_cache[top_k][passage_idx]
            for idx, score in semantic_candidates:
                update_candidate(self.passage_node_keys[idx], score)
        else:
            scores = np.dot(self.passage_embeddings, seed_vec.T)
            top_indices = np.argsort(scores)[-(top_k + 1):][::-1]
            for idx in top_indices:
                idx = int(idx)
                update_candidate(self.passage_node_keys[idx], float(scores[idx]))

        if self.use_evidence_paragraph_graph:
            shared_fact_counts = defaultdict(int)
            for fact_key in getattr(self, "chunk_to_fact_node_ids", {}).get(passage_key, set()):
                for candidate_key in getattr(self, "fact_node_to_chunk_ids", {}).get(fact_key, set()):
                    if candidate_key != passage_key:
                        shared_fact_counts[candidate_key] += 1

            shared_entity_counts = defaultdict(int)
            for entity_key in getattr(self, "chunk_to_entity_node_ids", {}).get(passage_key, set()):
                support = getattr(self, "ent_node_to_chunk_ids", {}).get(entity_key, set())
                if len(support) > self.evidence_paragraph_max_entity_support:
                    continue
                for candidate_key in support:
                    if candidate_key != passage_key:
                        shared_entity_counts[candidate_key] += 1

            evidence_candidates = sorted(
                set(shared_fact_counts).union(shared_entity_counts),
                key=lambda key: (
                    shared_fact_counts.get(key, 0),
                    shared_entity_counts.get(key, 0),
                    key,
                ),
                reverse=True,
            )[:self.evidence_paragraph_max_candidates]
            evidence_indices = [
                self.passage_key_to_idx.get(candidate_key)
                for candidate_key in evidence_candidates
            ]
            valid_evidence_indices = [
                int(idx) for idx in evidence_indices if idx is not None
            ]
            if scores is None and valid_evidence_indices:
                evidence_base_scores = np.dot(
                    self.passage_embeddings[valid_evidence_indices],
                    seed_vec.T,
                )
                evidence_score_by_idx = {
                    idx: float(score)
                    for idx, score in zip(
                        valid_evidence_indices,
                        np.asarray(evidence_base_scores).reshape(-1),
                    )
                }
            else:
                evidence_score_by_idx = {}

            for candidate_key, candidate_idx in zip(
                    evidence_candidates, evidence_indices):
                fact_bonus = min(shared_fact_counts.get(candidate_key, 0), 4) * 0.08
                entity_bonus = min(shared_entity_counts.get(candidate_key, 0), 6) * 0.03
                if candidate_idx is None:
                    continue
                candidate_idx = int(candidate_idx)
                base_score = (
                    float(scores[candidate_idx])
                    if scores is not None
                    else evidence_score_by_idx[candidate_idx]
                )
                evidence_score = base_score + fact_bonus + entity_bonus
                update_candidate(candidate_key, evidence_score)

        neighbors = sorted(
            candidate_scores.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )[:top_k]

        self.paragraph_neighbor_cache[cache_key] = neighbors
        return neighbors

    def _update_paragraph_agent_info(
            self,
            passage_info: Dict[str, Dict],
            passage_key: str,
            score: float,
            path: List[str],
            source: str = "paragraph_agent",
            evidence_score: Optional[float] = None,
            mean_policy_probability: float = 0.0,
            mean_query_similarity: Optional[float] = None,
            seed_score: float = 0.0) -> None:
        if passage_key in getattr(self, "base_retrieval_masked_passages", set()):
            return
        current = passage_info.get(passage_key)
        if current is None or score > current.get("score", -float("inf")):
            hops = max(0, len(path) - 1)
            passage_info[passage_key] = {
                "score": float(score),
                "search_score": float(score),
                "evidence_score": float(
                    score if evidence_score is None else evidence_score
                ),
                "mean_policy_probability": float(mean_policy_probability),
                "mean_query_similarity": float(
                    evidence_score
                    if mean_query_similarity is None and evidence_score is not None
                    else (mean_query_similarity or 0.0)
                ),
                "seed_score": float(seed_score),
                "search_hops": int(hops),
                "path": list(path),
                "source": source,
                "agent": "paragraph",
                "confidence": max(
                    0.0,
                    float(score if evidence_score is None else evidence_score),
                ),
            }

    def run_paragraph_graph_search(
            self,
            query: str,
            seed_passages: Optional[List[Any]] = None,
            query_emb: Optional[torch.Tensor] = None) -> Dict[str, Dict]:
        """Run the paragraph agent on an implicit paragraph-only KNN graph."""
        if self.paragraph_policy_network is None:
            self.load_rl_agent(training=False)
        if query_emb is None:
            query_emb = self._get_query_embedding_tensor(query)

        paragraph_top_k = getattr(self.global_config, "paragraph_agent_seed_top_k", 20)
        dpr_ids, dpr_scores = self.dense_passage_retrieval(query)
        dpr_scores_norm = min_max_normalize(dpr_scores)

        seed_mode = self.paragraph_seed_mode
        if seed_mode not in {"hybrid", "dpr_only", "fact_only"}:
            raise ValueError(f"Unsupported paragraph seed mode: {seed_mode}")
        fact_seed_score_mode = self.paragraph_fact_seed_score_mode
        if fact_seed_score_mode not in {"fixed", "normalized"}:
            raise ValueError(
                f"Unsupported Fact seed score mode: {fact_seed_score_mode}"
            )

        fact_seed_candidates = []
        for seed in seed_passages or []:
            if isinstance(seed, (tuple, list)) and len(seed) >= 2:
                fact_seed_candidates.append((seed[0], float(seed[1])))
            else:
                fact_seed_candidates.append((seed, 0.5))
        dpr_seed_candidates = [
            (self.passage_node_keys[int(doc_id)], float(score))
            for doc_id, score in zip(
                dpr_ids[:paragraph_top_k],
                dpr_scores_norm[:paragraph_top_k],
            )
            if self.passage_node_keys[int(doc_id)]
            not in getattr(self, "base_retrieval_masked_passages", set())
        ]
        fact_seed_keys = {key for key, _ in fact_seed_candidates}
        dpr_seed_keys = {key for key, _ in dpr_seed_candidates}

        seeds = []
        seen = set()
        selected_fact = 0
        selected_dpr = 0
        if seed_mode != "dpr_only":
            for passage_key, normalized_score in fact_seed_candidates:
                if (
                        passage_key in self.passage_key_to_idx
                        and passage_key not in seen
                        and passage_key not in getattr(
                            self,
                            "base_retrieval_masked_passages",
                            set(),
                        )):
                    seed_score = (
                        normalized_score
                        if fact_seed_score_mode == "normalized"
                        else 0.5
                    )
                    seeds.append((passage_key, float(seed_score)))
                    seen.add(passage_key)
                    selected_fact += 1

        if seed_mode != "fact_only":
            for passage_key, score in dpr_seed_candidates:
                if passage_key not in seen:
                    seeds.append((passage_key, float(score)))
                    seen.add(passage_key)
                    selected_dpr += 1

        stats = self.last_paragraph_search_stats
        if stats:
            stats["queries"] += 1
            stats["fact_seed_candidates"] += len(fact_seed_candidates)
            stats["dpr_seed_candidates"] += len(dpr_seed_candidates)
            stats["seed_overlap"] += len(fact_seed_keys.intersection(dpr_seed_keys))
            stats["selected_fact_seeds"] += selected_fact
            stats["selected_dpr_seeds"] += selected_dpr
            stats["selected_total_seeds"] += len(seeds)

        if not seeds:
            return {}

        beam_width = getattr(self.global_config, "paragraph_agent_beam_width", 20)
        max_steps = getattr(self.global_config, "paragraph_agent_max_steps", 2)
        passage_info = {}
        active_beams = []

        seed_similarities = []
        for passage_key, _ in seeds:
            passage_emb = self.get_embedding_for_node(passage_key)
            seed_similarities.append(
                F.cosine_similarity(
                    query_emb.view(1, -1), passage_emb.view(1, -1)
                ) if passage_emb is not None else query_emb.new_zeros(1)
            )
            if not getattr(self, "optimize_static_graph_ops", False):
                seed_similarities[-1] = [seed_similarities[-1].item()]
        seed_scores = (
            self._search_scores_to_cpu(seed_similarities)
            if getattr(self, "optimize_static_graph_ops", False) else seed_similarities
        )
        for (passage_key, score), similarity in zip(seeds, seed_scores):
            path = [passage_key]
            seed_query_similarity = similarity[0]
            active_beams.append({
                "path": path,
                "score": float(score),
                "seed_score": float(score),
                "policy_logprob_sum": 0.0,
                "query_similarity_sum": float(seed_query_similarity),
            })
            self._update_paragraph_agent_info(
                passage_info,
                passage_key,
                float(score),
                path,
                evidence_score=float(seed_query_similarity),
                mean_query_similarity=float(seed_query_similarity),
                seed_score=float(score),
            )

        with torch.no_grad():
            for _ in range(max_steps):
                candidates = []
                scored_beams = []
                for beam in active_beams:
                    current_key = beam["path"][-1]
                    neighbors = [
                        key for key, _ in self._get_paragraph_neighbors(current_key)
                        if key not in beam["path"]
                    ]
                    if not neighbors:
                        continue

                    current_emb = self.get_embedding_for_node(current_key)
                    if current_emb is None:
                        continue
                    neighbor_embs = []
                    valid_neighbors = []
                    for neighbor_key in neighbors:
                        emb = self.get_embedding_for_node(neighbor_key)
                        if emb is not None:
                            neighbor_embs.append(emb)
                            valid_neighbors.append(neighbor_key)
                    if not valid_neighbors:
                        continue

                    history_embs = [
                        self.get_embedding_for_node(node_key)
                        for node_key in beam["path"]
                    ]
                    history_embs = [emb for emb in history_embs if emb is not None]
                    if not history_embs:
                        continue

                    dist = self.paragraph_policy_network(
                        current_emb.view(1, -1),
                        query_emb,
                        torch.stack(history_embs).unsqueeze(0),
                        torch.stack(neighbor_embs)
                    )
                    if dist.probs.nelement() == 0:
                        continue

                    neighbor_tensor = torch.stack(neighbor_embs)
                    query_scores = F.cosine_similarity(
                        query_emb.expand(neighbor_tensor.shape[0], -1),
                        neighbor_tensor,
                        dim=1,
                    )
                    if getattr(self, "optimize_static_graph_ops", False):
                        scored_beams.append((beam, valid_neighbors, dist.logits, query_scores))
                    else:
                        scored_beams.append((beam, valid_neighbors,
                                             dist.logits.detach().cpu().tolist(),
                                             query_scores.detach().cpu().tolist()))
                flat_scores = [
                    tensor for _, _, policy, similarity in scored_beams
                    for tensor in (policy, similarity)
                ]
                score_groups = (
                    self._search_scores_to_cpu(flat_scores)
                    if getattr(self, "optimize_static_graph_ops", False) else flat_scores
                )
                for index, (beam, valid_neighbors, _, _) in enumerate(scored_beams):
                    policy_scores, query_scores = score_groups[2 * index:2 * index + 2]
                    for neighbor_key, policy_score, query_score in zip(
                            valid_neighbors, policy_scores, query_scores):
                        new_score = beam["score"] + 0.5 * policy_score + query_score
                        new_path = beam["path"] + [neighbor_key]
                        hops = len(new_path) - 1
                        policy_logprob_sum = (
                            float(beam.get("policy_logprob_sum", 0.0))
                            + float(policy_score)
                        )
                        query_similarity_sum = (
                            float(beam.get("query_similarity_sum", 0.0))
                            + float(query_score)
                        )
                        mean_policy_probability = math.exp(
                            policy_logprob_sum / max(1, hops)
                        )
                        mean_query_similarity = (
                            query_similarity_sum / float(hops + 1)
                        )
                        # This bounded confidence describes the evidence itself;
                        # the cumulative score remains dedicated to beam search.
                        evidence_score = (
                            0.50 * float(query_score)
                            + 0.30 * mean_query_similarity
                            + 0.20 * mean_policy_probability
                        )
                        candidate = {
                            "path": new_path,
                            "score": new_score,
                            "seed_score": float(beam.get("seed_score", 0.0)),
                            "policy_logprob_sum": policy_logprob_sum,
                            "query_similarity_sum": query_similarity_sum,
                        }
                        candidates.append(candidate)
                        self._update_paragraph_agent_info(
                            passage_info,
                            neighbor_key,
                            new_score,
                            new_path,
                            evidence_score=evidence_score,
                            mean_policy_probability=mean_policy_probability,
                            mean_query_similarity=mean_query_similarity,
                            seed_score=float(beam.get("seed_score", 0.0)),
                        )

                if not candidates:
                    break
                active_beams = sorted(
                    candidates,
                    key=lambda item: (item["score"], tuple(item["path"])),
                    reverse=True,
                )[:beam_width]

        return passage_info

    def _should_trigger_repair(self, *args, **kwargs):
        """Compatibility hook: repair is unavailable in this release."""
        return False

    @staticmethod
    def _repair_normalize_entity(value: Any) -> str:
        value = text_processing(str(value or ""))
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @classmethod
    def _repair_entity_aliases(cls, value: Any) -> Set[str]:
        normalized = cls._repair_normalize_entity(value)
        if not normalized:
            return set()
        aliases = {normalized}
        without_type = re.sub(
            r"\s+(film|movie|song|album|director|actor|actress|politician|writer|company)$",
            "",
            normalized,
        ).strip()
        if without_type:
            aliases.add(without_type)
        without_generation = re.sub(
            r"\s+(?:[ivxlcdm]+|\d+(?:st|nd|rd|th))$",
            "",
            normalized,
        ).strip()
        if without_generation:
            aliases.add(without_generation)
        return aliases

    @staticmethod
    def _repair_looks_like_date(value: Any) -> bool:
        normalized = text_processing(str(value or ""))
        month = r"january|february|march|april|may|june|july|august|september|october|november|december"
        return bool(re.search(rf"\b(?:{month}|\d{{3,4}})\b", normalized))

    @classmethod
    def _repair_relation_family(cls, relation: Any, query: str = "", tail: Any = "") -> str:
        rel = cls._repair_normalize_entity(relation)
        query_norm = cls._repair_normalize_entity(query)
        if any(operator in rel for operator in (
            "is earlier than", "is later than", "is older than", "is younger than",
            "same as", "greater than", "less than",
        )):
            return "comparison_operator"
        comparison = any(token in query_norm for token in (
            "earlier", "later", "older", "younger", "born first", "died first", "lived longer"
        ))
        if "direct" in rel:
            return "director"
        if "compos" in rel or "music by" in rel or "musical score by" in rel:
            return "composer"
        if "spouse" in rel or "married" in rel or "husband" in rel or "wife" in rel:
            return "spouse"
        if "grandfather" in rel:
            return "grandfather"
        if re.search(r"\bfather\b", rel):
            return "father"
        if re.search(r"\bmother\b", rel):
            return "mother"
        if "death" in rel or "died" in rel:
            if "place" in rel or (" in" in f" {rel}" and not cls._repair_looks_like_date(tail)):
                return "death_place"
            return "death_date"
        if "birth" in rel or "born" in rel:
            if comparison or "date" in rel or " on" in f" {rel}" or cls._repair_looks_like_date(tail):
                return "birth_date"
            return "birth_place"
        if "release" in rel or "premiere" in rel:
            return "release_date"
        if "national" in rel or "country" in rel or "is from" in rel:
            return "country"
        if "study" in rel or "educated" in rel or "attended" in rel:
            return "education"

        stopwords = {"a", "an", "the", "is", "was", "were", "of", "in", "on", "at", "by", "has", "had", "to"}
        tokens = []
        for token in rel.split():
            if token in stopwords:
                continue
            for suffix in ("ing", "ed", "es", "s"):
                if len(token) > len(suffix) + 2 and token.endswith(suffix):
                    token = token[:-len(suffix)]
                    break
            tokens.append(token)
        return "generic:" + " ".join(tokens)



    def _get_raw_query_triples(self, query: str) -> List[List[str]]:
        cached = getattr(self, "query_triplets_cache", {}).get(query, {})
        if isinstance(cached, dict) and isinstance(cached.get("raw_triples"), list):
            return [triple for triple in cached["raw_triples"] if isinstance(triple, list) and len(triple) >= 3]
        self.extract_query_triplets(query)
        cached = getattr(self, "query_triplets_cache", {}).get(query, {})
        if isinstance(cached, dict) and isinstance(cached.get("raw_triples"), list):
            return [triple for triple in cached["raw_triples"] if isinstance(triple, list) and len(triple) >= 3]
        return []




    def _merge_agent_passage_info(
            self,
            fact_info: Dict[str, Dict],
            paragraph_info: Dict[str, Dict],
            repair_info: Dict[str, Dict]) -> Dict[str, Dict]:
        merged: Dict[str, Dict] = {}
        repair_candidate_only = bool(getattr(self.global_config, "repair_candidate_only", False))
        source_weights = {
            "fact": getattr(self.global_config, "fact_agent_weight", 0.55),
            "paragraph": getattr(self.global_config, "paragraph_agent_weight", 0.35),
            "repair": getattr(self.global_config, "repair_agent_weight", 0.25),
        }

        def add_items(items: Dict[str, Dict], source_key: str) -> None:
            for passage_key, info in items.items():
                raw_score = float(info.get("score", 0.0))
                path_length = max(1, len(info.get("path", []) or []) - 1)
                score = raw_score + source_weights[source_key]
                current = merged.get(passage_key)
                if current is None:
                    merged[passage_key] = dict(info)
                    merged[passage_key]["score"] = score
                    merged[passage_key]["sources"] = {source_key}
                    merged[passage_key]["source_scores"] = {source_key: raw_score}
                    merged[passage_key]["source_path_lengths"] = {source_key: path_length}
                    merged[passage_key]["source_details"] = {
                        source_key: {
                            "search_score": float(info.get("search_score", raw_score)),
                            "evidence_score": float(info.get("evidence_score", raw_score)),
                            "mean_policy_probability": float(
                                info.get("mean_policy_probability", 0.0)
                            ),
                            "mean_query_similarity": float(
                                info.get("mean_query_similarity", 0.0)
                            ),
                            "seed_score": float(info.get("seed_score", 0.0)),
                            "search_hops": int(info.get("search_hops", path_length)),
                        }
                    }
                else:
                    previous_score = current.get("score", -float("inf"))
                    current["score"] = max(current.get("score", -float("inf")), score)
                    current.setdefault("sources", set()).add(source_key)
                    current.setdefault("source_scores", {})[source_key] = raw_score
                    current.setdefault("source_path_lengths", {})[source_key] = path_length
                    current.setdefault("source_details", {})[source_key] = {
                        "search_score": float(info.get("search_score", raw_score)),
                        "evidence_score": float(info.get("evidence_score", raw_score)),
                        "mean_policy_probability": float(
                            info.get("mean_policy_probability", 0.0)
                        ),
                        "mean_query_similarity": float(
                            info.get("mean_query_similarity", 0.0)
                        ),
                        "seed_score": float(info.get("seed_score", 0.0)),
                        "search_hops": int(info.get("search_hops", path_length)),
                    }
                    current["confidence"] = max(current.get("confidence", 0.0), info.get("confidence", 0.0))
                    if score >= previous_score:
                        current["path"] = info.get("path", current.get("path", []))

        add_items(fact_info or {}, "fact")
        add_items(paragraph_info or {}, "paragraph")
        if repair_candidate_only:
            for passage_key in (repair_info or {}):
                if passage_key not in merged:
                    merged[passage_key] = {
                        "score": 0.0,
                        "path": [],
                        "sources": set(),
                        "confidence": 0.0,
                        "evidence_nodes": [],
                        "ppr_score": 0.0,
                        "candidate_sources": ["repair"],
                        "source_scores": {},
                        "source_path_lengths": {},
                        "source_details": {},
                    }
        else:
            add_items(repair_info or {}, "repair")
        for info in merged.values():
            if isinstance(info.get("sources"), set):
                info["sources"] = sorted(info["sources"])
        return merged

    def _load_query_conditioned_ranker(self) -> None:
        if not self.query_conditioned_ranker_path:
            raise ValueError(
                "rerank_fusion_mode=query_conditioned requires "
                "query_conditioned_ranker_path"
            )
        checkpoint = torch.load(
            self.query_conditioned_ranker_path,
            map_location="cpu",
            weights_only=True,
        )
        input_dim = int(checkpoint["input_dim"])
        hidden_dim = int(checkpoint["hidden_dim"])
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.query_conditioned_ranker = model
        self.query_conditioned_ranker_mean = (
            checkpoint["feature_mean"].detach().cpu().float()
        )
        self.query_conditioned_ranker_std = (
            checkpoint["feature_std"].detach().cpu().float()
        )
        self.query_conditioned_ranker_feature_selector = (
            checkpoint["feature_selector"].detach().cpu().long()
        )
        self.query_conditioned_ranker_feature_schema = str(
            checkpoint.get("feature_schema", "legacy_v1")
        )
        logger.info(
            "Loaded query-conditioned candidate ranker: %s "
            "(input=%d, hidden=%d, schema=%s)",
            self.query_conditioned_ranker_path,
            input_dim,
            hidden_dim,
            self.query_conditioned_ranker_feature_schema,
        )

    @staticmethod
    def _set_aware_terms(value: Any) -> Set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
            "was", "were", "with",
        }
        return {
            token for token in re.findall(r"[a-z0-9]+", str(value).lower())
            if token not in stopwords
        }

    def _load_set_aware_selector(self) -> None:
        checkpoint = torch.load(
            self.set_aware_selector_path,
            map_location="cpu",
            weights_only=True,
        )
        input_dim = int(checkpoint["input_dim"])
        if input_dim not in {39, 51, 53}:
            raise ValueError(
                "Set-aware selector expects semantic 39-d, QueryPlan 51-d or progress 53-d "
                f"schema, got {input_dim}"
            )
        uses_progress = bool(checkpoint.get("uses_selection_progress_features", False))
        if (input_dim == 53) != uses_progress:
            raise ValueError("53-d selection progress schema requires matching checkpoint metadata")
        uses_plan = bool(checkpoint.get("uses_query_plan_features", input_dim in {51, 53}))
        if uses_progress and not uses_plan:
            raise ValueError("Selection progress schema requires QueryPlan features")
        uses_dynamic_plan = dynamic_query_plan_enabled(checkpoint)
        model = nn.Sequential(
            nn.Linear(input_dim, int(checkpoint["hidden_dim"])),
            nn.ReLU(),
            nn.Linear(int(checkpoint["hidden_dim"]), 1),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.set_aware_selector = model
        self.set_aware_selector_mean = checkpoint["feature_mean"].detach().cpu().float()
        self.set_aware_ablate_query_plan_features = bool(checkpoint.get("ablated_query_plan_features", False))
        self.set_aware_selector_std = checkpoint["feature_std"].detach().cpu().float()
        self.set_aware_selector_mode = str(checkpoint.get("selector_mode", "residual"))
        self.set_aware_uses_query_plan = uses_plan
        self.set_aware_uses_selection_progress = uses_progress
        self.set_aware_uses_dynamic_query_plan = uses_dynamic_plan
        self.set_aware_bridge_candidate_limit = int(
            checkpoint.get("bridge_candidate_limit", 0)
        )

        if os.path.isfile(self.openie_results_path):
            self._load_set_aware_openie_metadata()
        logger.info(
            "Loaded set-aware semantic selector: %s (bridge_top_k=%d)",
            self.set_aware_selector_path,
            self.set_aware_bridge_candidate_limit,
        )

    def _load_set_aware_openie_metadata(self) -> None:
        with open(self.openie_results_path, encoding="utf-8") as file:
            payload = json.load(file)
        documents = payload.get("docs", []) if isinstance(payload, dict) else payload
        metadata = {}
        for document in documents:
            passage = str(document.get("passage", ""))
            entities = {
                " ".join(re.findall(r"[a-z0-9]+", str(entity).lower()))
                for entity in document.get("extracted_entities", [])
            }
            relations = {
                " ".join(re.findall(r"[a-z0-9]+", str(triple[1]).lower()))
                for triple in document.get("extracted_triples", [])
                if len(triple) >= 3
            }
            entities.discard("")
            relations.discard("")
            metadata[str(document["idx"])] = {
                "title": passage.split("\n", 1)[0].strip().lower(),
                "text": passage.lower(),
                "entities": entities,
                "relations": relations,
                "tokens": self._set_aware_terms(passage),
                "triples": valid_openie_triples(document.get("extracted_triples", [])),
            }
        self.set_aware_openie_metadata = metadata
        logger.info(
            "Loaded set-aware OpenIE metadata: passages=%d",
            len(metadata),
        )

    def _set_aware_semantic_features(
            self, passage_keys: List[str], anchor_index: int) -> np.ndarray:
        if not self.set_aware_openie_metadata:
            self._load_set_aware_openie_metadata()
        metadata = self.set_aware_openie_metadata
        anchor = metadata.get(str(passage_keys[anchor_index]), {})
        anchor_entities = anchor.get("entities", set())
        anchor_relations = anchor.get("relations", set())
        anchor_tokens = anchor.get("tokens", set())
        anchor_title = anchor.get("title", "")
        rows = []
        for passage_key in passage_keys:
            candidate = metadata.get(str(passage_key), {})
            entities = candidate.get("entities", set())
            relations = candidate.get("relations", set())
            tokens = candidate.get("tokens", set())
            shared_entities = len(anchor_entities & entities)
            entity_union = len(anchor_entities | entities)
            relation_union = len(anchor_relations | relations)
            token_union = len(anchor_tokens | tokens)
            token_jaccard = len(anchor_tokens & tokens) / max(1, token_union)
            bridge = float(shared_entities > 0)
            rows.append((
                min(shared_entities, 5) / 5.0,
                shared_entities / max(1, entity_union),
                len(anchor_relations & relations) / max(1, relation_union),
                token_jaccard,
                float(bool(anchor_title) and anchor_title in candidate.get("text", "")),
                float(bool(candidate.get("title")) and candidate["title"] in anchor.get("text", "")),
                bridge,
                bridge * (1.0 - token_jaccard),
            ))
        return np.asarray(rows, dtype=np.float32)

    def _apply_set_aware_selection(
            self,
            candidate_components: List[Dict],
            unary_scores: Dict[str, float],
            query: str) -> List[str]:
        candidate_pool = [
            record for record in candidate_components
            if record["passage"] in unary_scores
        ]
        if len(candidate_pool) < 2:
            return sorted(
                unary_scores,
                key=lambda passage: (unary_scores[passage], passage),
                reverse=True,
            )

        # These are the same 14 candidate features used by the offline exporter.
        full_rankings = self._component_source_rankings(candidate_pool)

        def normalized_source(key: str) -> Tuple[Dict[str, float], float]:
            present = [
                float(record[key]) for record in candidate_pool
                if record[key] is not None
            ]
            if not present:
                return ({record["passage"]: 0.0 for record in candidate_pool}, 0.0)
            low, high = min(present), max(present)
            values = {
                record["passage"]: (
                    (float(record[key]) - low) / (high - low)
                    if record[key] is not None and high - low > 1e-12
                    else (1.0 if record[key] is not None else 0.0)
                )
                for record in candidate_pool
            }
            ranked = sorted(values.values(), reverse=True)
            return values, float(ranked[0] - ranked[1]) if len(ranked) > 1 else 1.0

        fact, fact_gap = normalized_source("fact")
        paragraph, paragraph_gap = normalized_source("paragraph")
        similarity, similarity_gap = normalized_source("similarity")
        reciprocal = {
            source: {
                passage: 1.0 / float(rank)
                for rank, passage in enumerate(ranking, start=1)
            }
            for source, ranking in full_rankings.items()
        }
        base = []
        passage_keys = []
        tail_scores = []
        for record in candidate_pool:
            passage = record["passage"]
            fact_present = float(record["fact"] is not None)
            paragraph_present = float(record["paragraph"] is not None)
            source_count = fact_present + paragraph_present
            passage_keys.append(passage)
            tail_scores.append(float(unary_scores[passage]))
            base.append((
                float(record["similarity"]), source_count,
                max(0.0, source_count - 1.0), fact_present, paragraph_present,
                fact[passage], paragraph[passage], similarity[passage],
                fact[passage] * fact_gap, paragraph[passage] * paragraph_gap,
                similarity[passage] * similarity_gap,
                reciprocal["fact"].get(passage, 0.0),
                reciprocal["paragraph"].get(passage, 0.0),
                reciprocal["similarity"].get(passage, 0.0),
            ))
        base = np.asarray(base, dtype=np.float32)
        tail_scores = np.asarray(tail_scores, dtype=np.float32)
        scores = (
            base[:, 0].copy()
            if self.set_aware_anchor_mode == "dpr" else tail_scores
        )
        order = np.argsort(-scores, kind="stable")
        anchor = int(order[0])
        eligible = np.ones(len(order), dtype=bool)
        if self.set_aware_bridge_candidate_limit > 0:
            eligible[:] = False
            eligible[order[:self.set_aware_bridge_candidate_limit]] = True
        columns = np.asarray((0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13))
        candidate = base[:, columns]

        raw_query_triples = self._get_raw_query_triples(query)
        plan_scores = (
            query_plan_slot_scores(
                passage_keys, self.set_aware_openie_metadata, raw_query_triples
            )
            if self.set_aware_uses_query_plan else None
        )
        plan_cache = {}
        self.set_aware_last_query_plan_trace = []

        def conditional(reference: int) -> np.ndarray:
            reference_rows = np.broadcast_to(candidate[reference], candidate.shape)
            pair = np.column_stack((
                base[:, 3] * base[reference, 3],
                base[:, 4] * base[reference, 4],
                base[:, 3] * base[reference, 4] + base[:, 4] * base[reference, 3],
                np.abs(base[:, 5] - base[reference, 5]),
                np.abs(base[:, 6] - base[reference, 6]),
                np.abs(base[:, 7] - base[reference, 7]),
                np.abs(scores - scores[reference]),
            ))
            parts = [
                candidate,
                reference_rows,
                scores.reshape(-1, 1),
                np.full((len(scores), 1), scores[reference], dtype=np.float32),
                pair,
                self._set_aware_semantic_features(passage_keys, reference),
            ]
            if self.set_aware_uses_query_plan:
                parts.append(plan_features)
            features = np.concatenate(parts, axis=1).astype(np.float32)
            if self.set_aware_uses_selection_progress:
                features = append_selection_progress(features, len(selected), budget)
            return features

        budget = 2
        if self.set_aware_selection_mode in {
                "dynamic_setmax", "dynamic_setmax_recovery"}:
            budget = min(4, max(2, len(self._get_raw_query_triples(query))))
            if self.set_aware_selection_mode == "dynamic_setmax_recovery":
                budget = min(4, budget + 1)
        selected = [anchor]
        while len(selected) < budget:
            candidate_mask = eligible.copy()
            candidate_mask[selected] = False
            candidate_indices = np.flatnonzero(candidate_mask)
            if not len(candidate_indices):
                break
            references = (
                selected if self.set_aware_selection_mode in {
                    "dynamic_setmax", "dynamic_setmax_recovery"
                }
                else selected[-1:]
            )
            reference_scores = []
            if self.set_aware_uses_query_plan:
                trace = {}
                plan_features = query_plan_state_features(
                    passage_keys, self.set_aware_openie_metadata, raw_query_triples, selected,
                    dynamic=getattr(self, "set_aware_uses_dynamic_query_plan", False),
                    static_scores=plan_scores, cache=plan_cache, trace=trace,
                )
                if trace:
                    self.set_aware_last_query_plan_trace.append(dict(selected=list(selected), **trace))
            for reference in references:
                features = conditional(reference)
                normalized = (
                    torch.from_numpy(features[candidate_indices]).float()
                    - self.set_aware_selector_mean
                ) / self.set_aware_selector_std
                if getattr(self, "set_aware_ablate_query_plan_features", False):
                    normalized[:, 39:51] = 0
                with torch.no_grad():
                    logits = self.set_aware_selector(normalized).view(-1).numpy()
                reference_scores.append(logits)
            bridge_scores = np.max(reference_scores, axis=0)
            if self.set_aware_selector_mode == "residual":
                prior = scores[candidate_mask]
                scale = float(prior.std())
                bridge_scores += (
                    (prior - float(prior.mean())) / scale
                    if scale > 1e-6 else np.zeros_like(prior)
                )
            selected.append(int(candidate_indices[int(np.argmax(bridge_scores))]))
        tail_order = np.argsort(-tail_scores, kind="stable")
        selected_set = set(selected)
        selected.extend(
            int(idx) for idx in tail_order if int(idx) not in selected_set
        )
        return [passage_keys[idx] for idx in selected]

    @staticmethod
    def _rank_component_indices(
            values: np.ndarray,
            available: Optional[np.ndarray] = None) -> List[int]:
        indices = np.arange(len(values))
        if available is not None:
            indices = indices[available]
        return sorted(
            indices.tolist(),
            key=lambda idx: (float(values[idx]), idx),
            reverse=True,
        )

    @staticmethod
    def _component_source_rankings(
            candidate_components: List[Dict]) -> Dict[str, List[str]]:
        rankings = {}
        for source in ("fact", "paragraph", "similarity", "repair"):
            rankings[source] = [
                record["passage"]
                for record in sorted(
                    (
                        item for item in candidate_components
                        if (
                            item[source] is not None
                            and (source != "repair" or item[source] > 0.0)
                        )
                    ),
                    key=lambda item: (item[source], item["passage"]),
                    reverse=True,
                )
            ]
        return rankings

    @classmethod
    def _multiscale_ranker_query_features(
            cls,
            candidate_matrix: np.ndarray,
            include_top20: bool) -> np.ndarray:
        source_indices = {
            "fact": np.flatnonzero(candidate_matrix[:, 3] > 0.5),
            "paragraph": np.flatnonzero(candidate_matrix[:, 4] > 0.5),
            "similarity": np.arange(len(candidate_matrix)),
        }
        score_columns = {"fact": 5, "paragraph": 6, "similarity": 7}
        orders = {}
        features = []
        for source in ("fact", "paragraph", "similarity"):
            column = score_columns[source]
            indices = source_indices[source]
            order = sorted(
                indices.tolist(),
                key=lambda idx: (float(candidate_matrix[idx, column]), idx),
                reverse=True,
            )
            orders[source] = order
            ranked = candidate_matrix[order, column]
            features.extend((
                float(np.mean(ranked[:10])) if len(ranked) else 0.0,
                float(ranked[4] - ranked[5]) if len(ranked) > 5 else 0.0,
                float(ranked[9] - ranked[10]) if len(ranked) > 10 else 0.0,
            ))
            if include_top20:
                features.append(
                    float(np.mean(ranked[:20])) if len(ranked) else 0.0
                )
        for left, right in (
                ("fact", "similarity"),
                ("fact", "paragraph"),
                ("paragraph", "similarity")):
            for k in (2, 5, 10):
                features.append(
                    len(set(orders[left][:k]).intersection(orders[right][:k]))
                    / float(k)
                )
        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _paragraph_evidence_ranker_features(
            candidate_components: List[Dict]) -> np.ndarray:
        """Expose non-accumulating Paragraph evidence signals to the ranker."""
        rows = []
        for record in candidate_components:
            present = record.get("paragraph_evidence") is not None
            rows.append([
                float(record.get("paragraph_evidence") or 0.0),
                float(record.get("paragraph_policy_probability") or 0.0),
                float(record.get("paragraph_query_path_mean") or 0.0),
                float(record.get("paragraph_seed_score") or 0.0),
                min(1.0, float(record.get("paragraph_hops") or 0.0) / 2.0)
                if present else 0.0,
            ])
        return np.asarray(rows, dtype=np.float32)

    def _query_conditioned_candidate_scores(
            self,
            candidate_components: List[Dict]) -> Dict[str, float]:
        if self.query_conditioned_ranker is None:
            raise RuntimeError("Query-conditioned candidate ranker is not loaded")

        full_rankings = self._component_source_rankings(candidate_components)
        selected = set(full_rankings["fact"][:self.query_conditioned_fact_budget])
        selected.update(
            full_rankings["paragraph"][:self.query_conditioned_paragraph_budget]
        )
        if self.query_conditioned_similarity_budget > 0:
            selected.update(
                full_rankings["similarity"][:self.query_conditioned_similarity_budget]
            )
        if self.query_conditioned_repair_budget > 0:
            selected.update(
                full_rankings["repair"][:self.query_conditioned_repair_budget]
            )
        candidate_pool = [
            record for record in candidate_components
            if record["passage"] in selected
        ]
        if not candidate_pool:
            return {}

        def normalized_source(key: str) -> Tuple[Dict[str, float], float]:
            present = [
                float(record[key]) for record in candidate_pool
                if record[key] is not None
            ]
            if not present:
                return ({record["passage"]: 0.0 for record in candidate_pool}, 0.0)
            low, high = min(present), max(present)
            values = {
                record["passage"]: (
                    (float(record[key]) - low) / (high - low)
                    if record[key] is not None and high - low > 1e-12
                    else (1.0 if record[key] is not None else 0.0)
                )
                for record in candidate_pool
            }
            ranked = sorted(values.values(), reverse=True)
            gap = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
            return values, float(gap)

        fact, fact_gap = normalized_source("fact")
        paragraph, paragraph_gap = normalized_source("paragraph")
        similarity, similarity_gap = normalized_source("similarity")
        pool_rankings = self._component_source_rankings(candidate_pool)
        reciprocal_ranks = {
            source: {
                passage: 1.0 / float(rank)
                for rank, passage in enumerate(ranking, start=1)
            }
            for source, ranking in pool_rankings.items()
        }
        candidate_features = []
        for record in candidate_pool:
            passage = record["passage"]
            fact_present = float(record["fact"] is not None)
            paragraph_present = float(record["paragraph"] is not None)
            source_count = fact_present + paragraph_present
            candidate_features.append([
                float(record["similarity"]),
                source_count,
                max(0.0, source_count - 1.0),
                fact_present,
                paragraph_present,
                fact[passage],
                paragraph[passage],
                similarity[passage],
                fact[passage] * fact_gap,
                paragraph[passage] * paragraph_gap,
                similarity[passage] * similarity_gap,
                reciprocal_ranks["fact"].get(passage, 0.0),
                reciprocal_ranks["paragraph"].get(passage, 0.0),
                reciprocal_ranks["similarity"].get(passage, 0.0),
            ])

        passages = [record["passage"] for record in candidate_components]
        fact_raw = np.asarray([
            float(record["fact"] or 0.0) for record in candidate_components
        ])
        paragraph_raw = np.asarray([
            float(record["paragraph"] or 0.0) for record in candidate_components
        ])
        similarity_raw = np.asarray([
            float(record["similarity"] or 0.0) for record in candidate_components
        ])
        agreement = np.asarray([
            float(record["agreement"]) for record in candidate_components
        ])
        available = {
            "fact": np.asarray([
                record["fact"] is not None for record in candidate_components
            ]),
            "paragraph": np.asarray([
                record["paragraph"] is not None for record in candidate_components
            ]),
            "similarity": np.asarray([
                record["similarity"] is not None for record in candidate_components
            ]),
        }
        query_features = []

        def add_source(values: np.ndarray, source_available: np.ndarray) -> None:
            ranked_values = [
                float(values[idx])
                for idx in self._rank_component_indices(values, source_available)
            ]
            padded = ranked_values[:3] + [0.0] * max(0, 3 - len(ranked_values))
            query_features.extend(padded[:3])
            query_features.append(padded[0] - padded[1])
            query_features.append(
                float(np.mean(ranked_values[:5])) if ranked_values else 0.0
            )
            query_features.append(
                float(len(ranked_values)) / max(1.0, float(len(values)))
            )

        add_source(fact_raw, available["fact"])
        add_source(paragraph_raw, available["paragraph"])
        add_source(similarity_raw, available["similarity"])
        query_features.extend((0.0, 0.0, 0.0))
        fact_top = set(self._rank_component_indices(
            fact_raw, available["fact"]
        )[:2])
        paragraph_top = set(self._rank_component_indices(
            paragraph_raw, available["paragraph"]
        )[:2])
        similarity_top = set(self._rank_component_indices(
            similarity_raw, available["similarity"]
        )[:2])
        query_features.extend((
            len(fact_top.intersection(similarity_top)) / 2.0,
            len(fact_top.intersection(paragraph_top)) / 2.0,
            len(paragraph_top.intersection(similarity_top)) / 2.0,
            float(np.mean(agreement)) if len(agreement) else 0.0,
            0.0,
            min(1.0, float(len(passages)) / 200.0),
        ))

        candidate_matrix = np.asarray(candidate_features, dtype=np.float32)
        query_vector = np.asarray(query_features, dtype=np.float32)
        full_features = np.concatenate((
            candidate_matrix,
            np.broadcast_to(
                query_vector,
                (len(candidate_pool), len(query_vector)),
            ),
        ), axis=1)
        selector = self.query_conditioned_ranker_feature_selector.numpy()
        selected_features = full_features[:, selector]
        schema = self.query_conditioned_ranker_feature_schema
        if schema.startswith("paragraph_evidence_v1"):
            selected_features = np.concatenate((
                selected_features,
                self._paragraph_evidence_ranker_features(candidate_pool),
            ), axis=1)
        if "multiscale_k2_5_10" in schema:
            multiscale = self._multiscale_ranker_query_features(
                candidate_matrix,
                include_top20=("multiscale_k2_5_10_20" in schema),
            )
            selected_features = np.concatenate((
                selected_features,
                np.broadcast_to(
                    multiscale,
                    (len(candidate_pool), len(multiscale)),
                ),
            ), axis=1)
        if selected_features.shape[1] != self.query_conditioned_ranker_mean.numel():
            raise ValueError(
                "Query-conditioned feature dimension mismatch: "
                f"built={selected_features.shape[1]}, "
                f"checkpoint={self.query_conditioned_ranker_mean.numel()}, "
                f"schema={schema}"
            )
        normalized = (
            torch.from_numpy(selected_features).float()
            - self.query_conditioned_ranker_mean
        ) / self.query_conditioned_ranker_std
        with torch.no_grad():
            scores = self.query_conditioned_ranker(normalized).view(-1).tolist()
        return {
            record["passage"]: float(score)
            for record, score in zip(candidate_pool, scores)
        }

    def _build_candidate_components(
            self,
            passage_info: Dict[str, Dict],
            query_emb_tensor: torch.Tensor,
            ppr_scores_dict: Optional[Dict[str, float]] = None) -> List[Dict]:
        components = []
        ppr_scores_dict = ppr_scores_dict or {}
        passages = list(passage_info)
        similarity_scores = {}
        similarity_batch_size = 4096
        for start in range(0, len(passages), similarity_batch_size):
            batch_passages = passages[start:start + similarity_batch_size]
            valid_passages = []
            passage_embeddings = []
            for passage in batch_passages:
                passage_emb_tensor = self.get_embedding_for_node(passage)
                if passage_emb_tensor is None:
                    continue
                valid_passages.append(passage)
                passage_embeddings.append(passage_emb_tensor)
            if not passage_embeddings:
                continue
            embedding_batch = torch.stack(passage_embeddings)
            query_batch = query_emb_tensor.view(1, -1).expand(
                embedding_batch.shape[0], -1
            )
            batch_scores = F.cosine_similarity(
                query_batch,
                embedding_batch,
                dim=1,
            ).detach().cpu().tolist()
            similarity_scores.update(zip(valid_passages, batch_scores))

        for passage, info in passage_info.items():
            similarity_score = similarity_scores.get(passage)
            if similarity_score is None:
                continue

            ppr_score = (
                info.get("ppr_score", ppr_scores_dict.get(passage, 0.0))
                if self.use_ppr_rerank else 0.0
            )
            sources = info.get("sources", [])
            source_agreement = max(0, len(sources) - 1) if isinstance(sources, list) else 0
            source_scores = info.get("source_scores", {})
            source_path_lengths = info.get("source_path_lengths", {})
            paragraph_details = info.get("source_details", {}).get("paragraph", {})

            def adjusted_source_score(source_key: str) -> Optional[float]:
                if source_key not in source_scores:
                    return None
                score = float(source_scores[source_key])
                if self.normalize_path_by_length:
                    score /= max(1, int(source_path_lengths.get(source_key, 1)))
                return score

            components.append({
                "passage": passage,
                "info": info,
                "path": float(info.get("score", 0.0)),
                "fact": adjusted_source_score("fact"),
                "paragraph": adjusted_source_score("paragraph"),
                "paragraph_search": (
                    float(paragraph_details.get("search_score"))
                    if paragraph_details else None
                ),
                "paragraph_evidence": (
                    float(paragraph_details.get("evidence_score"))
                    if paragraph_details else None
                ),
                "paragraph_policy_probability": (
                    float(paragraph_details.get("mean_policy_probability", 0.0))
                    if paragraph_details else None
                ),
                "paragraph_query_path_mean": (
                    float(paragraph_details.get("mean_query_similarity", 0.0))
                    if paragraph_details else None
                ),
                "paragraph_seed_score": (
                    float(paragraph_details.get("seed_score", 0.0))
                    if paragraph_details else None
                ),
                "paragraph_hops": (
                    float(paragraph_details.get("search_hops", 0.0))
                    if paragraph_details else None
                ),
                "similarity": float(similarity_score),
                "ppr": float(ppr_score),
                "repair": float(info.get("confidence", 0.0)),
                "agreement": float(source_agreement),
            })
        return components

    @staticmethod
    def _normalize_candidate_components(candidate_components: List[Dict]) -> Dict[str, Dict[str, float]]:
        def minmax_component(key: str, missing_value: float = 0.0) -> Dict[str, float]:
            present = [record[key] for record in candidate_components if record[key] is not None]
            if not present:
                return {record["passage"]: missing_value for record in candidate_components}
            low, high = min(present), max(present)
            if high - low <= 1e-12:
                return {
                    record["passage"]: (1.0 if record[key] is not None else missing_value)
                    for record in candidate_components
                }
            return {
                record["passage"]: (
                    (record[key] - low) / (high - low)
                    if record[key] is not None else missing_value
                )
                for record in candidate_components
            }

        normalized = {
            key: minmax_component(key)
            for key in ("fact", "paragraph", "similarity", "ppr")
        }

        def top_gap(key: str) -> float:
            values = sorted(
                (normalized[key][record["passage"]] for record in candidate_components),
                reverse=True,
            )
            return float(values[0] - values[1]) if len(values) > 1 else (1.0 if values else 0.0)

        gaps = {key: top_gap(key) for key in ("fact", "paragraph", "similarity")}
        for record in candidate_components:
            passage = record["passage"]
            record["info"]["coordinator_normalized"] = {
                "fact": normalized["fact"][passage],
                "paragraph": normalized["paragraph"][passage],
                "similarity": normalized["similarity"][passage],
                "ppr": normalized["ppr"][passage],
                "fact_gap": gaps["fact"],
                "paragraph_gap": gaps["paragraph"],
                "similarity_gap": gaps["similarity"],
            }
        return normalized

    def _rank_fusion_candidate_scores(
            self,
            candidate_components: List[Dict]) -> Dict[str, float]:
        """Fuse source-local ranks without comparing heterogeneous raw scores."""
        passages = [record["passage"] for record in candidate_components]
        records_by_passage = {
            record["passage"]: record for record in candidate_components
        }
        rrf_k = self.rank_fusion_rrf_k

        def source_rank_scores(key: str) -> Dict[str, float]:
            ranked = sorted(
                (
                    record for record in candidate_components
                    if record[key] is not None
                ),
                key=lambda record: (record[key], record["passage"]),
                reverse=True,
            )
            scale = float(rrf_k + 1)
            return {
                record["passage"]: scale / float(rrf_k + rank)
                for rank, record in enumerate(ranked, start=1)
            }

        fact_scores = source_rank_scores("fact")
        paragraph_scores = source_rank_scores("paragraph")
        similarity_scores = source_rank_scores("similarity")
        ppr_scores = source_rank_scores("ppr") if self.use_ppr_rerank else {}

        fused = {}
        for passage in passages:
            record = records_by_passage[passage]
            fused[passage] = float(
                self.w_fact_policy * fact_scores.get(passage, 0.0)
                + self.w_paragraph_policy * paragraph_scores.get(passage, 0.0)
                + self.w_sim * similarity_scores.get(passage, 0.0)
                + self.w_ppr * ppr_scores.get(passage, 0.0)
                + self.w_repair_confidence * record["repair"]
                + self.w_source_agreement * min(1.0, record["agreement"])
            )
            record["info"]["coordinator_rank_rrf"] = {
                "fact": fact_scores.get(passage, 0.0),
                "paragraph": paragraph_scores.get(passage, 0.0),
                "similarity": similarity_scores.get(passage, 0.0),
                "ppr": ppr_scores.get(passage, 0.0),
            }
        return fused

    def _candidate_feature_vector(
            self,
            passage: str,
            info: Dict,
            query_emb_tensor: torch.Tensor,
            ppr_scores_dict: Optional[Dict[str, float]] = None,
            heuristic_score: Optional[float] = None) -> Optional[np.ndarray]:
        passage_emb_tensor = self.get_embedding_for_node(passage)
        if passage_emb_tensor is None:
            return None

        path_score = float(info.get("score", 0.0))
        similarity_score = F.cosine_similarity(
            query_emb_tensor.view(1, -1),
            passage_emb_tensor.view(1, -1)
        ).item()
        ppr_scores_dict = ppr_scores_dict or {}
        ppr_score = float(info.get("ppr_score", ppr_scores_dict.get(passage, 0.0))) if self.use_ppr_rerank else 0.0
        confidence = float(info.get("confidence", 0.0))
        sources = info.get("sources", [])
        if isinstance(sources, set):
            sources = sorted(sources)
        if not isinstance(sources, list):
            sources = []
        source_count = float(len(sources))
        source_agreement = max(0.0, source_count - 1.0)
        path_len = float(len(info.get("path", []) or []))
        evidence_count = float(len(info.get("evidence_nodes", []) or []))
        coordinator_normalized = info.get("coordinator_normalized", {})
        normalized_fact = float(coordinator_normalized.get("fact", 0.0))
        normalized_paragraph = float(coordinator_normalized.get("paragraph", 0.0))
        normalized_similarity = float(coordinator_normalized.get("similarity", 0.0))
        fact_gap = float(coordinator_normalized.get("fact_gap", 0.0))
        paragraph_gap = float(coordinator_normalized.get("paragraph_gap", 0.0))
        similarity_gap = float(coordinator_normalized.get("similarity_gap", 0.0))
        if heuristic_score is None:
            heuristic_score = (
                self.w_policy * path_score
                + self.w_sim * similarity_score
                + self.w_ppr * ppr_score
                + self.w_repair_confidence * confidence
                + self.w_source_agreement * source_agreement
            )

        return np.array([
            np.clip(path_score, -20.0, 20.0),
            similarity_score,
            ppr_score,
            confidence,
            source_count,
            source_agreement,
            1.0 if "fact" in sources else 0.0,
            1.0 if "paragraph" in sources else 0.0,
            1.0 if "repair" in sources else 0.0,
            min(path_len, 20.0) / 20.0,
            min(evidence_count, 10.0) / 10.0,
            float(heuristic_score),
            normalized_fact,
            normalized_paragraph,
            normalized_similarity,
            normalized_fact * fact_gap,
            normalized_paragraph * paragraph_gap,
            normalized_similarity * similarity_gap,
            normalized_similarity * confidence,
        ], dtype=np.float32)

    def _score_candidate_with_coordinator(
            self,
            passage: str,
            info: Dict,
            query_emb_tensor: torch.Tensor,
            ppr_scores_dict: Optional[Dict[str, float]] = None,
            heuristic_score: Optional[float] = None) -> float:
        features = self._candidate_feature_vector(
            passage=passage,
            info=info,
            query_emb_tensor=query_emb_tensor,
            ppr_scores_dict=ppr_scores_dict,
            heuristic_score=heuristic_score,
        )
        if features is None:
            return -float("inf")
        if (
                not self.use_learned_coordinator
                or (
                    self.coordinator_weights is None
                    and self.coordinator_mlp_state is None
                )
                or self.coordinator_feature_mean is None
                or self.coordinator_feature_std is None):
            return float(features[11])

        normalized = (features - self.coordinator_feature_mean) / self.coordinator_feature_std
        if self.coordinator_mlp_state is not None:
            state = self.coordinator_mlp_state
            hidden = np.maximum(
                0.0,
                np.dot(normalized, state["hidden_weight"].T)
                + state["hidden_bias"],
            )
            learned_logit = float(
                np.dot(hidden, state["output_weight"])
                + state["output_bias"]
            )
        else:
            learned_logit = float(
                np.dot(normalized, self.coordinator_weights)
                + self.coordinator_bias
            )
        learned_delta = 1.0 / (1.0 + np.exp(-np.clip(learned_logit, -20.0, 20.0))) - 0.5
        return float(features[11] + self.coordinator_blend_alpha * learned_delta)

    def collect_coordinator_examples(
            self,
            queries: List[str],
            gold_docs: List[List[str]],
            max_queries: Optional[int] = None,
            max_negatives_per_query: int = 30,
            return_query_ids: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Collect passage-level features and binary labels for lightweight coordinator training."""
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()
        selected_queries = queries[:max_queries] if max_queries and max_queries > 0 else queries
        selected_gold_docs = gold_docs[:len(selected_queries)]
        self.precompute_embeddings(selected_queries)
        self._reset_repair_stats()

        features, labels, query_ids = [], [], []
        for query_idx, (query, gold_for_query) in enumerate(tqdm(
                list(zip(selected_queries, selected_gold_docs)),
                desc="Collecting coordinator examples")):
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts = self.rank_facts(query, query_fact_scores)
            query_emb_tensor = self._get_query_embedding_tensor(query)
            dpr_ids_for_repair, _ = self.dense_passage_retrieval(query)

            if top_k_facts:
                fact_info, query_emb_raw = self.run_rl_graph_search_rerank(
                    query,
                    top_k_facts,
                    top_k_fact_indices=top_k_fact_indices,
                    query_fact_scores=query_fact_scores
                )
                if isinstance(query_emb_raw, np.ndarray):
                    query_emb_tensor = torch.from_numpy(query_emb_raw).float().to(self.device).view(1, -1)
                else:
                    query_emb_tensor = query_emb_raw.to(self.device).view(1, -1)
            else:
                fact_info = {}

            ranked_fact_seed_items = sorted(
                fact_info.items(),
                key=lambda item: (
                    item[1].get("score", -float("inf")),
                    item[0],
                ),
                reverse=True,
            )[:getattr(self.global_config, "paragraph_agent_seed_top_k", 20)]
            ranked_fact_seed_items = self._gate_fact_seeds_for_paragraph(
                ranked_fact_seed_items,
                query_emb_tensor,
            )
            fact_seed_scores = (
                min_max_normalize(np.array([
                    float(info.get("score", 0.0))
                    for _, info in ranked_fact_seed_items
                ]))
                if ranked_fact_seed_items
                else np.array([], dtype=np.float32)
            )
            fact_seed_passages = [
                (passage_key, float(seed_score))
                for (passage_key, _), seed_score in zip(
                    ranked_fact_seed_items,
                    fact_seed_scores,
                )
            ]
            if getattr(self.global_config, "use_paragraph_agent", True):
                paragraph_info = self.run_paragraph_graph_search(
                    query=query,
                    seed_passages=fact_seed_passages,
                    query_emb=query_emb_tensor
                )
            else:
                paragraph_info = {}

            repair_info = {}
            if self._should_trigger_repair(
                    query,
                    fact_info,
                    paragraph_info=paragraph_info,
                    dpr_ids=dpr_ids_for_repair):
                active_paths = [
                    info.get("path", [])
                    for _, info in sorted(
                        fact_info.items(),
                        key=lambda item: (
                            item[1].get("score", -float("inf")),
                            item[0],
                        ),
                        reverse=True
                    )[:3]
                ]
                repair_info = self._repair_broken_path(
                    query=query,
                    top_k_facts=top_k_facts,
                    active_paths=active_paths
                )

            passage_info = self._merge_agent_passage_info(fact_info, paragraph_info, repair_info)
            candidate_components = self._build_candidate_components(
                passage_info=passage_info,
                query_emb_tensor=query_emb_tensor,
            )
            normalized_components = self._normalize_candidate_components(candidate_components)
            rank_fusion_scores = (
                self._rank_fusion_candidate_scores(candidate_components)
                if self.rerank_fusion_mode == "rank_rrf"
                else {}
            )
            components_by_passage = {
                record["passage"]: record for record in candidate_components
            }
            query_examples = []
            gold_set = set(gold_for_query or [])
            for passage, info in passage_info.items():
                record = components_by_passage.get(passage)
                if record is None:
                    continue
                if self.rerank_fusion_mode == "rank_rrf":
                    heuristic_score = rank_fusion_scores[passage]
                elif self.normalize_rerank_features:
                    heuristic_score = (
                        self.w_fact_policy * normalized_components["fact"][passage]
                        + self.w_paragraph_policy * normalized_components["paragraph"][passage]
                        + self.w_sim * normalized_components["similarity"][passage]
                        + self.w_ppr * normalized_components["ppr"][passage]
                        + self.w_repair_confidence * record["repair"]
                        + self.w_source_agreement * min(1.0, record["agreement"])
                    )
                else:
                    heuristic_score = (
                        self.w_policy * record["path"]
                        + self.w_sim * record["similarity"]
                        + self.w_ppr * record["ppr"]
                        + self.w_repair_confidence * record["repair"]
                        + self.w_source_agreement * record["agreement"]
                    )
                feat = self._candidate_feature_vector(
                    passage=passage,
                    info=info,
                    query_emb_tensor=query_emb_tensor,
                    heuristic_score=heuristic_score,
                )
                if feat is None:
                    continue
                try:
                    passage_content = self.chunk_embedding_store.get_row(passage)["content"]
                except Exception:
                    passage_content = ""
                label = 1.0 if passage_content in gold_set else 0.0
                query_examples.append((feat, label))

            positives = [(feat, label) for feat, label in query_examples if label > 0]
            negatives = [(feat, label) for feat, label in query_examples if label <= 0]
            negatives = sorted(negatives, key=lambda item: item[0][11], reverse=True)[:max_negatives_per_query]
            for feat, label in positives + negatives:
                features.append(feat)
                labels.append(label)
                query_ids.append(query_idx)

        if not features:
            empty_x = np.empty((0, 19), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.float32)
            if return_query_ids:
                return empty_x, empty_y, np.empty((0,), dtype=np.int64)
            return empty_x, empty_y
        x = np.vstack(features).astype(np.float32)
        y = np.array(labels, dtype=np.float32)
        if return_query_ids:
            return x, y, np.array(query_ids, dtype=np.int64)
        return x, y

    def _gate_fact_seeds_for_paragraph(
            self,
            ranked_fact_seed_items: List[Tuple[str, Dict[str, Any]]],
            query_emb: torch.Tensor) -> List[Tuple[str, Dict[str, Any]]]:
        """Apply the learned high-level KEEP/SKIP action to Paragraph seeds only."""
        if not getattr(self.global_config, "rl_paragraph_seed_gate_enabled", False):
            return list(ranked_fact_seed_items)

        threshold = float(getattr(
            self.global_config,
            "rl_paragraph_seed_gate_threshold",
            0.5,
        ))
        min_keep = max(0, int(getattr(
            self.global_config,
            "rl_paragraph_seed_gate_min_keep",
            0,
        )))
        stats = getattr(self, "last_fact_seed_gate_stats", None)
        if stats is None:
            stats = {
                "checked": 0,
                "kept": 0,
                "skipped": 0,
                "keep_probability_sum": 0.0,
            }
            self.last_fact_seed_gate_stats = stats

        scored_items = []
        query_emb = query_emb.to(self.device).view(1, -1)
        with torch.no_grad():
            for passage_key, info in ranked_fact_seed_items:
                path = info.get("path", [])
                start_node = path[0] if path else None
                seed_emb = self.get_embedding_for_node(start_node) if start_node else None
                if seed_emb is None:
                    keep_probability = 1.0
                else:
                    keep_logit = self.rl_policy_network.seed_keep_logit(
                        seed_emb.view(1, -1),
                        query_emb,
                    )
                    keep_probability = float(torch.sigmoid(keep_logit).item())

                stats["checked"] += 1
                stats["keep_probability_sum"] += keep_probability
                scored_items.append((passage_key, info, keep_probability))

        selected_keys = {
            passage_key
            for passage_key, _, probability in scored_items
            if probability >= threshold
        }
        if len(selected_keys) < min_keep:
            by_probability = sorted(
                scored_items,
                key=lambda item: (item[2], item[0]),
                reverse=True,
            )
            selected_keys.update(
                passage_key
                for passage_key, _, _ in by_probability[:min_keep]
            )

        kept_items = [
            (passage_key, info)
            for passage_key, info, _ in scored_items
            if passage_key in selected_keys
        ]
        stats["kept"] += len(kept_items)
        stats["skipped"] += len(scored_items) - len(kept_items)
        return kept_items

    def run_rl_graph_search_rerank(
            self,
            query: str,
            top_k_facts: List[Tuple],
            top_k_fact_indices: Optional[List[int]] = None,
            query_fact_scores: Optional[np.ndarray] = None,
            max_seeds: Optional[int] = None,
            max_steps: Optional[int] = None) -> Tuple[Dict, torch.Tensor]:
        """
        Executes RL-based beam search on the graph to find high-probability paths leading to relevant passages.
        """
        beam_width = self.beam_width

        # 1. Get ranked seeds. Training and inference should use the same seed policy.
        ranked_seeds = self.get_ranked_seed_nodes(
            top_k_facts=top_k_facts,
            top_k_fact_indices=top_k_fact_indices,
            query_fact_scores=query_fact_scores,
            max_seeds=(
                max_seeds
                if max_seeds is not None
                else getattr(self.global_config, "rl_seed_top_k", 12)
            ),
            include_fact_nodes=True,
        )
        
        # 2. Prepare Query Embedding
        query_emb_obj = self.query_to_embedding['passage'].get(query, None)
        if query_emb_obj is None:
            query_emb_obj = self.embedding_model.batch_encode(query, 
                                                             instruction=get_query_instruction('query_to_passage'), 
                                                             norm=True)
            self.query_to_embedding['passage'][query] = query_emb_obj

        if isinstance(query_emb_obj, np.ndarray):
            query_emb = torch.from_numpy(query_emb_obj).float().to(self.device)
        else:
            query_emb = query_emb_obj.to(self.device)
        query_emb = query_emb.view(1, -1)

        # 3. Validate seeds and initialize beam
        valid_seeds = ranked_seeds
        
        if not valid_seeds:
            logger.warning("No valid RL starting seeds found, trying CoEviRAG repair before DPR.")
            repaired = self._repair_broken_path(query=query, top_k_facts=top_k_facts, active_paths=[])
            if repaired:
                return self._finalize_fact_passages(
                    repaired, query_emb, include_masked=True), query_emb
            dpr_fallback = {}
            dpr_ids, dpr_scores = self.dense_passage_retrieval(query)
            for doc_id, score in zip(dpr_ids, dpr_scores):
                node_key = self.passage_node_keys[doc_id]
                dpr_fallback[node_key] = {'score': float(score), 'path': [], 'source': 'dpr'}
            return dpr_fallback, query_emb

        seed_score_weight = getattr(self.global_config, "rl_seed_score_weight", 0.5)
        active_beams = [
            {'path': [seed], 'score': float(seed_score) * seed_score_weight}
            for seed, seed_score in valid_seeds
        ]
        passage_info = defaultdict(lambda: {'score': -float('inf'), 'path': []})
        for beam in active_beams:
            self._update_passage_info_from_path(passage_info, beam['path'], beam['score'])

        # --- 4. Beam Search Loop ---
        search_steps = (
            max_steps
            if max_steps is not None
            else self.rl_max_episode_steps
        )
        for step in range(search_steps):
            if not active_beams: break
            
            all_candidate_beams = []
            expanded_nodes_in_step = set()

            beam_inputs = []
            for beam in active_beams:
                current_node = beam['path'][-1]
                if current_node in expanded_nodes_in_step: continue
                expanded_nodes_in_step.add(current_node)

                try:
                    neighbor_names = [
                        name for name in self._get_fact_search_neighbors(
                            current_node,
                            query,
                            previous_node=(
                                beam["path"][-2]
                                if len(beam["path"]) > 1 else None
                            ),
                        )
                        if name not in beam['path']
                    ]
                    if not neighbor_names:
                        continue

                    neighbor_embs_list = []
                    valid_neighbor_names = []
                    for neighbor_name in neighbor_names:
                        emb = self.get_embedding_for_node(neighbor_name)
                        if emb is not None:
                            neighbor_embs_list.append(emb)
                            valid_neighbor_names.append(neighbor_name)
                    if not valid_neighbor_names:
                        continue

                    current_node_emb = self.get_embedding_for_node(current_node)
                    if current_node_emb is None:
                        continue
                    history_embs_list = [
                        self.get_embedding_for_node(node_name)
                        for node_name in beam['path']
                    ]
                    valid_history_embs = [
                        emb for emb in history_embs_list if emb is not None
                    ]
                    if not valid_history_embs:
                        continue
                    beam_inputs.append({
                        "beam": beam,
                        "neighbor_names": valid_neighbor_names,
                        "neighbor_embs": torch.stack(neighbor_embs_list),
                        "current_emb": current_node_emb.view(1, -1),
                        "history_embs": torch.stack(valid_history_embs).unsqueeze(0),
                    })
                except Exception as e:
                    logger.debug(f"Beam Search Error (Node: {current_node}): {e}")
                    continue

            with torch.no_grad():
                history_lengths = {
                    item["history_embs"].shape[1] for item in beam_inputs
                }
                use_batched_scoring = (
                    self.batch_fact_beam_scoring
                    and len(beam_inputs) > 1
                    and len(history_lengths) == 1
                )
                if use_batched_scoring:
                    policy_score_groups = self.rl_policy_network.action_log_probs_batch(
                        torch.cat(
                            [item["current_emb"] for item in beam_inputs],
                            dim=0,
                        ),
                        query_emb.expand(len(beam_inputs), -1),
                        torch.cat(
                            [item["history_embs"] for item in beam_inputs],
                            dim=0,
                        ),
                        [item["neighbor_embs"] for item in beam_inputs],
                    )
                    policy_score_groups = self._search_scores_to_cpu(policy_score_groups)
                else:
                    policy_score_groups = []
                    for item in beam_inputs:
                        dist = self.rl_policy_network(
                            item["current_emb"],
                            query_emb,
                            item["history_embs"],
                            item["neighbor_embs"],
                        )
                        policy_score_groups.append(dist.logits)
                    policy_score_groups = self._search_scores_to_cpu(policy_score_groups)

            for item, policy_scores in zip(beam_inputs, policy_score_groups):
                beam = item["beam"]
                prefix_evidence = (
                    self._path_evidence_state(beam["path"], beam["score"])
                    if getattr(self, "reuse_fact_path_prefix", False) else None
                )
                batch_mapping = (
                    getattr(self, "batch_fact_path_mapping", False)
                    and prefix_evidence is not None
                )
                extensions = []
                for next_node, policy_score in zip(
                        item["neighbor_names"], policy_scores):
                    new_path_score = beam['score'] + policy_score
                    new_path = beam['path'] + [next_node]
                    extension = {
                        'path': new_path,
                        'score': new_path_score,
                    }
                    all_candidate_beams.append(extension)
                    if batch_mapping:
                        extensions.append(extension)
                    else:
                        self._update_passage_info_from_path(
                            passage_info,
                            new_path,
                            new_path_score,
                            prefix_evidence=prefix_evidence,
                        )
                if batch_mapping:
                    self._map_fact_path_extensions(passage_info, extensions, prefix_evidence)
            
            if not all_candidate_beams:
                active_paths = [beam['path'] for beam in active_beams]
                repaired = self._repair_broken_path(query=query, top_k_facts=top_k_facts, active_paths=active_paths)
                passage_info.update(repaired)
                break
            
            # Pruning
            active_beams = sorted(
                all_candidate_beams,
                key=lambda item: (item["score"], tuple(item["path"])),
                reverse=True,
            )[:beam_width]

        # --- 5. Result check ---
        if not passage_info:
            logger.warning("Fact navigation found no mapped paragraphs, trying CoEviRAG repair before DPR.")
            repaired = self._repair_broken_path(
                query=query,
                top_k_facts=top_k_facts,
                active_paths=[beam['path'] for beam in active_beams]
            )
            if repaired:
                return self._finalize_fact_passages(
                    repaired, query_emb, include_masked=True), query_emb

            logger.warning("CoEviRAG repair found no verified evidence, falling back to DPR.")
            dpr_fallback = {}
            dpr_ids, dpr_scores = self.dense_passage_retrieval(query)
            for doc_id, score in zip(dpr_ids, dpr_scores):
                node_key = self.passage_node_keys[doc_id]
                dpr_fallback[node_key] = {'score': float(score), 'path': [], 'source': 'dpr'}
            return dpr_fallback, query_emb
            
        return self._finalize_fact_passages(passage_info, query_emb), query_emb

    def rank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """
        Ranks candidate facts based on initial similarity scores to select top-k facts for linking.
        """
        link_top_k: int = self.global_config.linking_top_k

        rank_cache_key = (query, int(link_top_k), len(self.fact_node_keys))
        if self.cache_static_retrieval:
            cached_rank = self.fact_rank_cache.get(rank_cache_key)
            if cached_rank is not None:
                self.static_retrieval_cache_stats["fact_rank_hits"] += 1
                return cached_rank
            self.static_retrieval_cache_stats["fact_rank_misses"] += 1
        
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for ranking. Returning empty lists.")
            return [], [], {'facts_before_rank': [], 'facts_after_rank': []}
            
        try:
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            top_k_fact_indices = candidate_fact_indices
            top_k_facts = candidate_facts
                     
            result = (top_k_fact_indices, top_k_facts)
            if self.cache_static_retrieval:
                self.fact_rank_cache[rank_cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error in rank_facts: {str(e)}")
            return [], []

    def analyze_multihop_capability(self, test_set_path: str, dataset_name: str = "musique"):
        """
        Specialized analysis function that loads a test set, groups items by hop count, and evaluates Recall@10.
        """
        logger.info(f"Starting Multi-hop Capability analysis (Dataset: {dataset_name})...")
        
        with open(test_set_path, 'r') as f:
            data = json.load(f)
            
        hop_groups = {'2-hop': [], '3-hop': [], '4-hop': []}
        
        for item in data:
            num_hops = 0
            if 'contexts' in item: # Musique format
                num_hops = sum(1 for p in item['contexts'] if p.get('is_supporting', False))
            elif 'supporting_facts' in item: # 2Wiki/Hotpot format
                num_hops = len(item['supporting_facts'])
            
            if num_hops == 2:
                hop_groups['2-hop'].append(item)
            elif num_hops == 3:
                hop_groups['3-hop'].append(item)
            elif num_hops >= 4:
                hop_groups['4-hop'].append(item)
        
        logger.info(f"Grouping Statistics: 2-hop: {len(hop_groups['2-hop'])}, 3-hop: {len(hop_groups['3-hop'])}, 4-hop: {len(hop_groups['4-hop'])}")
            
        if self.rl_agent is None:
            self.load_rl_agent(training=False)
        
        results = {}
        all_queries = [item['question'] for item in data]
        self.precompute_embeddings(all_queries)
        
        evaluator = RetrievalRecall(global_config=self.global_config)
        
        for hop, items in hop_groups.items():
            if not items: continue
            logger.info(f"Evaluating {hop} ({len(items)} samples)...")
            
            queries = [i['question'] for i in items]
            
            gold_docs = []
            for sample in items:
                if 'supporting_facts' in sample:
                    gold_title = set([item[0] for item in sample['supporting_facts']])
                    gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
                    if dataset_name and dataset_name.startswith('hotpotqa'):
                        gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
                    else:
                        gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
                elif 'contexts' in sample:
                    gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
                else:
                    assert 'paragraphs' in sample, "`paragraphs` should be in sample"
                    gold_paragraphs = [p for p in sample['paragraphs'] if p.get('is_supporting', True)]
                    gold_doc = [item['title'] + '\n' + (item.get('text', item.get('paragraph_text'))) for item in gold_paragraphs]
                gold_docs.append(list(set(gold_doc)))

            retrieved_results = self.retrieve(queries=queries, num_to_retrieve=10)
            
            score_dict, _ = evaluator.calculate_metric_scores(
                gold_docs=gold_docs, 
                retrieved_docs=[r.docs for r in retrieved_results], 
                k_list = [1, 2, 5, 10]
            )
            results[hop] = score_dict
            
            logger.info(f">>> {hop} Result: {results[hop]}")

        logger.info("\n=== Final Hop Analysis Results ===")
        logger.info(json.dumps(results, indent=2))
        return results    

    # ==============================================================================
    #  Base Functionality (Indexing, Graph Ops, QA)
    # ==============================================================================

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.
        """
        graph_variant = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(getattr(self.global_config, "graph_variant", "")).strip(),
        ).strip("_.")
        graph_suffix = f"_{graph_variant}" if graph_variant else ""
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"coevirag_fact_graph{graph_suffix}.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def _run_openie_batch_with_retries(self, batch_rows):
        attempts = max(1, int(getattr(
            self.global_config, "openie_batch_retry_attempts", 3
        )))
        pending_rows = dict(batch_rows)
        merged_ner_results = {}
        merged_triple_results = {}
        error_messages = []
        for attempt in range(1, attempts + 1):
            ner_results, triple_results = self.openie.batch_openie(pending_rows)
            failed_ids = set()
            error_messages = []
            for chunk_id in pending_rows:
                ner_result = ner_results.get(chunk_id)
                triple_result = triple_results.get(chunk_id)
                chunk_errors = [
                    result.metadata.get("error")
                    for result in (ner_result, triple_result)
                    if result is not None and result.metadata.get("error")
                ]
                if ner_result is None or triple_result is None or chunk_errors:
                    failed_ids.add(chunk_id)
                    error_messages.extend(chunk_errors or [f"missing result for {chunk_id}"])
                    continue
                merged_ner_results[chunk_id] = ner_result
                merged_triple_results[chunk_id] = triple_result

            if not failed_ids:
                return merged_ner_results, merged_triple_results
            if attempt < attempts:
                delay = 10 * attempt
                logger.warning(
                    "OpenIE batch attempt %d/%d left %d/%d passages with API "
                    "errors; retrying only those passages in %ds.",
                    attempt,
                    attempts,
                    len(failed_ids),
                    len(batch_rows),
                    delay,
                )
                pending_rows = {
                    chunk_id: batch_rows[chunk_id] for chunk_id in failed_ids
                }
                time.sleep(delay)
        raise RuntimeError(
            f"OpenIE batch failed after {attempts} attempts with "
            f"{len(error_messages)} API errors; checkpoint was not written."
        )

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            batch_size = max(1, int(getattr(self.global_config, "openie_batch_size", 200)))
            new_items = list(new_openie_rows.items())
            max_new_passages = max(0, int(getattr(
                self.global_config, "openie_max_new_passages", 0
            )))
            all_new_items_count = len(new_items)
            if max_new_passages:
                new_items = new_items[:max_new_passages]
            for start in range(0, len(new_items), batch_size):
                batch_rows = dict(new_items[start:start + batch_size])
                new_ner_results_dict, new_triple_results_dict = self._run_openie_batch_with_retries(batch_rows)
                self.merge_openie_results(
                    all_openie_info,
                    batch_rows,
                    new_ner_results_dict,
                    new_triple_results_dict,
                )
                if self.global_config.save_openie:
                    self.save_openie_results(all_openie_info)
                logger.info(
                    "OpenIE checkpoint: %d/%d new passages processed.",
                    min(start + batch_size, len(new_items)),
                    len(new_items),
                )

            if len(new_items) < all_new_items_count:
                logger.info(
                    "OpenIE passage limit reached: processed %d of %d pending passages; "
                    "skipping graph construction for this partial run.",
                    len(new_items),
                    all_new_items_count,
                )
                return

        if self.global_config.save_openie and not chunk_keys_to_process:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the CoEviRAG framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            batch_size = max(1, int(getattr(self.global_config, "openie_batch_size", 200)))
            new_items = list(new_openie_rows.items())
            max_new_passages = max(0, int(getattr(
                self.global_config, "openie_max_new_passages", 0
            )))
            all_new_items_count = len(new_items)
            if max_new_passages:
                new_items = new_items[:max_new_passages]
            for start in range(0, len(new_items), batch_size):
                batch_rows = dict(new_items[start:start + batch_size])
                new_ner_results_dict, new_triple_results_dict = self._run_openie_batch_with_retries(batch_rows)
                self.merge_openie_results(
                    all_openie_info,
                    batch_rows,
                    new_ner_results_dict,
                    new_triple_results_dict,
                )
                if self.global_config.save_openie:
                    self.save_openie_results(all_openie_info)
                logger.info(
                    "OpenIE checkpoint: %d/%d new passages processed.",
                    min(start + batch_size, len(new_items)),
                    len(new_items),
                )

            if len(new_items) < all_new_items_count:
                logger.info(
                    "OpenIE passage limit reached: processed %d of %d pending passages; "
                    "skipping graph construction for this partial run.",
                    len(new_items),
                    all_new_items_count,
                )
                return

        if self.global_config.save_openie and not chunk_keys_to_process:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        expected_graph_nodes = len(set(entity_nodes)) + len(set(facts))
        if (
                not self.global_config.force_index_from_scratch
                and os.path.exists(self._graph_pickle_filename)
                and self.graph.vcount() == expected_graph_nodes):
            logger.info(
                f"Reusing existing CoEviRAG fact graph with {self.graph.vcount()} nodes and {self.graph.ecount()} edges."
            )
            return

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        self.graph = ig.Graph(directed=self.global_config.is_directed_graph)
        self.add_synonymy_edges()
        self.augment_graph()
        self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the CoEviRAG class.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.
        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.
        """
        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '
            context = ''.join(
                f'Wikipedia Title: {passage}\n\n' for passage in retrieved_passages
            ).rstrip()
            qa_prompt_name = qa_prompt_for_model(
                self.global_config.llm_name,
                self.global_config.dataset,
                self.prompt_template_manager,
            )
            all_qa_messages.append(
                self.prompt_template_manager.render(
                    name=qa_prompt_name,
                    prompt_user=prompt_user,
                    context=context,
                    question=query_solution.question,
                )
            )

        all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            pred_ans = parse_qa_answer(response_content)

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact-level edges from given triples to the graph.

        CoEviRAG keeps paragraph nodes out of the OpenIE KG. Each extracted
        triple becomes an explicit fact node:

            head_entity -- fact_node(head, relation, tail) -- tail_entity

        The supporting paragraph ids are stored in side indexes instead of
        being inserted as graph nodes.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to CoEviRAG fact graph.")

        self.fact_node_to_chunk_ids = getattr(self, "fact_node_to_chunk_ids", {})
        self.fact_node_to_triple = getattr(self, "fact_node_to_triple", {})
        self.chunk_to_fact_node_ids = getattr(self, "chunk_to_fact_node_ids", {})

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()
            facts_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)
                    if len(triple) != 3:
                        continue

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))
                    fact_key = compute_mdhash_id(content=str(triple), prefix=("fact-"))

                    self.node_to_node_stats[(node_key, fact_key)] = self.node_to_node_stats.get(
                        (node_key, fact_key), 0.0) + 1
                    self.node_to_node_stats[(fact_key, node_2_key)] = self.node_to_node_stats.get(
                        (fact_key, node_2_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)
                    facts_in_chunk.add(fact_key)
                    self.fact_node_to_triple[fact_key] = triple
                    self.fact_node_to_chunk_ids[fact_key] = self.fact_node_to_chunk_ids.get(
                        fact_key, set()).union(set([chunk_key]))

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

                self.chunk_to_fact_node_ids[chunk_key] = self.chunk_to_fact_node_ids.get(
                    chunk_key, set()).union(facts_in_chunk)

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.
        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        max_neighbors = max(
            0, int(getattr(self.global_config, "synonymy_edge_max_neighbors", 100))
        )
        candidate_map = {}

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if (
                            score < self.global_config.synonymy_edge_sim_threshold
                            or num_nns >= max_neighbors):
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        synonyms.append((nn, score))
                        num_nns += 1

            candidate_map[node_key] = synonyms

        mutual_knn = bool(getattr(
            self.global_config, "synonymy_edge_mutual_knn", False
        ))
        candidate_sets = {
            node_key: {neighbor for neighbor, _ in neighbors}
            for node_key, neighbors in candidate_map.items()
        }
        num_synonym_edges = 0
        for node_key, neighbors in candidate_map.items():
            for neighbor, score in neighbors:
                if mutual_knn and node_key not in candidate_sets.get(neighbor, set()):
                    continue
                self.node_to_node_stats[(node_key, neighbor)] = score
                num_synonym_edges += 1
        logger.info(
            "Added %d synonym edges (max_neighbors=%d, threshold=%.3f, mutual_knn=%s).",
            num_synonym_edges,
            max_neighbors,
            self.global_config.synonymy_edge_sim_threshold,
            mutual_knn,
        )

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.
        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds entity and fact nodes to the CoEviRAG fact graph.

        Paragraphs remain outside the OpenIE KG and are retrieved through
        fact/entity-to-paragraph indexes.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        fact_to_row = self.fact_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(fact_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes_external"] = len(set(passage_nodes_keys))

        fact_nodes_keys = self.fact_embedding_store.get_all_ids()
        graph_info["num_fact_nodes"] = len(set(fact_nodes_keys))

        # get # of total nodes
        graph_info["num_total_graph_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_fact_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        graph_info['num_triples_with_passage_node'] = 0

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] * 2

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes.
        """

        logger.info("Preparing for fast retrieval.")

        self.fact_score_cache.clear()
        self.fact_rank_cache.clear()
        self.dense_retrieval_cache.clear()
        self.fact_neighbor_cache.clear()
        self.paragraph_semantic_knn_cache.clear()
        self.fact_endpoint_cache.clear()
        self.evidence_semantic_neighbor_cache.clear()
        self.evidence_semantic_cache_signature = None
        for stat_name in self.static_retrieval_cache_stats:
            self.static_retrieval_cache_stats[stat_name] = 0

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.passage_key_to_idx = {
            key: idx for idx, key in enumerate(self.passage_node_keys)
        }
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.fact_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and fact nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_fact_nodes = [node_key for node_key in self.fact_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_fact_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_fact_nodes)} fact nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.fact_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.fact_node_keys if node_key in igraph_name_to_idx]
            self.passage_node_idxs = []
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.fact_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}
        self.fact_node_to_chunk_ids = {}
        self.fact_node_to_triple = {}
        self.chunk_to_fact_node_ids = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    fact_key = compute_mdhash_id(content=str(proc_triple), prefix="fact-")
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))
                    self.fact_node_to_chunk_ids[fact_key] = self.fact_node_to_chunk_ids.get(fact_key, set()).union(set([doc['idx']]))
                    self.fact_node_to_triple[fact_key] = proc_triple
                    self.chunk_to_fact_node_ids[doc['idx']] = self.chunk_to_fact_node_ids.get(doc['idx'], set()).union(set([fact_key]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.chunk_to_entity_node_ids = {}
        for entity_key, chunk_ids in (self.ent_node_to_chunk_ids or {}).items():
            for chunk_id in chunk_ids:
                self.chunk_to_entity_node_ids.setdefault(chunk_id, set()).add(entity_key)

        # Keep an immutable paragraph-side OpenIE view only when a relation gate uses it.
        relation_gate_enabled = bool(getattr(
            self.global_config, "repair_evidence_gate_relation_aware", False
        )) or int(getattr(
            self.global_config, "repair_evidence_gate_relation_top_k", 20
        )) > 0
        self.source_chunk_to_triples = {}
        if relation_gate_enabled:
            self.source_chunk_to_triples = {
                chunk_id: tuple(
                    self.fact_node_to_triple[fact_key]
                    for fact_key in sorted(fact_keys)
                    if fact_key in self.fact_node_to_triple
                )
                for chunk_id, fact_keys in self.chunk_to_fact_node_ids.items()
            }

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.
        """
        if self.cache_static_retrieval:
            cached_result = self.dense_retrieval_cache.get(query)
            if cached_result is not None:
                self.static_retrieval_cache_stats["dense_hits"] += 1
                return cached_result
            self.static_retrieval_cache_stats["dense_misses"] += 1

        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        result = (sorted_doc_ids, sorted_doc_scores)
        if self.cache_static_retrieval:
            self.dense_retrieval_cache[query] = result
        return result

    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    
