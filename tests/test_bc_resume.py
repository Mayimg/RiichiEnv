import torch

from riichienv_ml.models.transformer import TransformerPolicyNetwork
from riichienv_ml.utils import load_model_weights, load_torch_state_dict


def _small_model():
    return TransformerPolicyNetwork(
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_prog_len=8,
        max_cand_len=4,
        policy_head_type="cls",
    )


def test_load_torch_state_dict_supports_raw_and_wrapped_checkpoints(tmp_path):
    model = _small_model()
    raw_path = tmp_path / "raw.pth"
    wrapped_path = tmp_path / "wrapped.pth"

    torch.save(model.state_dict(), raw_path)
    torch.save({"state_dict": model.state_dict()}, wrapped_path)

    raw_state = load_torch_state_dict(str(raw_path))
    wrapped_state = load_torch_state_dict(str(wrapped_path))

    assert raw_state.keys() == wrapped_state.keys()


def test_load_model_weights_restores_saved_parameters(tmp_path):
    source = _small_model()
    target = _small_model()
    path = tmp_path / "source.pth"

    with torch.no_grad():
        for param in source.parameters():
            param.fill_(0.125)
        for param in target.parameters():
            param.zero_()

    torch.save(source.state_dict(), path)
    load_model_weights(target, str(path), map_location="cpu")

    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(source_param, target_param)
