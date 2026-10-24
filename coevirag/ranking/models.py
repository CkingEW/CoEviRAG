"""Checkpoint-compatible neural ranker construction and scoring."""
import numpy as np
import torch


def load_unary_ranker(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = torch.nn.Sequential(
        torch.nn.Linear(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])),
        torch.nn.ReLU(),
        torch.nn.Linear(int(checkpoint["hidden_dim"]), 1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def unary_scores(features, model, checkpoint, device):
    mean = checkpoint["feature_mean"].cpu().numpy()
    std = checkpoint["feature_std"].cpu().numpy()
    with torch.no_grad():
        return model(torch.from_numpy((features - mean) / std).float().to(device)).view(-1).cpu().numpy()


def candidate_unary_scores(data, indices, mode, model, checkpoint, device):
    if mode == "dpr":
        return data["base"][indices, 0].astype(np.float32)
    return unary_scores(data["features"][indices], model, checkpoint, device)


def load_selector(path, device, bridge_candidate_limit=None):
    if bridge_candidate_limit is not None and bridge_candidate_limit < 0:
        raise ValueError("bridge_candidate_limit must be nonnegative")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    from coevirag.release_protocol import validate_selector
    validate_selector(checkpoint)
    if bridge_candidate_limit is not None:
        checkpoint = {**checkpoint, "bridge_candidate_limit": bridge_candidate_limit}
    model = torch.nn.Sequential(
        torch.nn.Linear(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])),
        torch.nn.ReLU(),
        torch.nn.Linear(int(checkpoint["hidden_dim"]), 1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    schema = str(checkpoint.get("feature_schema", ""))
    selector_mode = "residual" if "residual" in schema else "direct"
    return model, checkpoint, selector_mode


def make_model(dim, device, residual=False):
    model = torch.nn.Sequential(torch.nn.Linear(dim, 16), torch.nn.ReLU(),
                                torch.nn.Linear(16, 1)).to(device)
    if residual:
        torch.nn.init.zeros_(model[2].weight)
        torch.nn.init.zeros_(model[2].bias)
    return model
