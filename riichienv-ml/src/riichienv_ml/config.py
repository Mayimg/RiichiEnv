from __future__ import annotations

import importlib
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from riichienv_ml.features.grp_agari_features import get_grp_input_dim

GAME_PARAMS = {
    4: {
        "tile_dim": 34,
        "num_actions": 82,
        "game_mode": "4p-red-half",
        "replay_rule": "mjsoul",
        "starting_scores": [25000, 25000, 25000, 25000],
    },
    3: {
        "tile_dim": 27,
        "num_actions": 60,
        "game_mode": "3p-red-half",
        "replay_rule": "mjsoul",
        "starting_scores": [35000, 35000, 35000],
    },
}


def import_class(dotted_path: str):
    """Dynamically import a class from a dotted path.

    e.g. "riichienv_ml.models.q_network.QNetwork" -> QNetwork class
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class GameConfig(BaseModel):
    n_players: Literal[3, 4] = 4
    replay_rule: Literal["tenhou", "mjsoul"] | None = None

    @model_validator(mode="after")
    def set_replay_rule_default(self) -> GameConfig:
        if self.replay_rule is None:
            self.replay_rule = GAME_PARAMS[self.n_players]["replay_rule"]
        return self

    @computed_field
    @property
    def tile_dim(self) -> int:
        return GAME_PARAMS[self.n_players]["tile_dim"]

    @computed_field
    @property
    def num_actions(self) -> int:
        return GAME_PARAMS[self.n_players]["num_actions"]

    @computed_field
    @property
    def game_mode(self) -> str:
        return GAME_PARAMS[self.n_players]["game_mode"]

    @computed_field
    @property
    def starting_scores(self) -> list[int]:
        return GAME_PARAMS[self.n_players]["starting_scores"]

    @computed_field
    @property
    def grp_input_dim(self) -> int:
        return get_grp_input_dim(self.n_players, self.replay_rule)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    in_channels: int = 74
    num_blocks: int = 8
    conv_channels: int = 128
    fc_dim: int = 256
    num_actions: int = 82
    tile_dim: int = 34
    aux_dims: int | None = None


class WandbConfig(BaseModel):
    wandb_entity: str = "smly"
    wandb_project: str = "riichienv"
    wandb_tags: list[str] = []
    wandb_group: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"


class OpponentConfig(BaseModel):
    config_path: str
    model_path: str


class EvaluatorConfig(BaseModel):
    model_path: str | None = None
    evaluator_name: str | None = None
    eval_episodes: int = 48
    eval_interval: int = 50000
    eval_device: str = "cpu"
    opponents: list[OpponentConfig] = []


class GrpConfig(WandbConfig):
    game: GameConfig = GameConfig()
    data_glob: str = "/data/mjsoul/mjsoul-4p/2024/**/*.jsonl.gz"
    val_data_glob: str = "/data/mjsoul/mjsoul-4p/2024/01/**/*.jsonl.gz"
    output: str = "grp_model.pth"
    device: str = "cuda"
    batch_size: int = 128
    num_workers: int = 12
    num_epochs: int = 10
    lr: float = 5e-4
    lr_eta_min: float = 1e-7
    samples_per_file: int = 32


class OfflineTrainConfig(WandbConfig):
    """Shared config for offline BC and CQL training."""

    game: GameConfig = GameConfig()
    data_glob: str = "/data/mjsoul/mjsoul-4p/2024/**/*.jsonl.gz"
    val_data_glob: str = ""
    grp_model: str | None = "./grp_model.pth"
    output: str = "bc_model.pth"
    device: str = "cuda"
    batch_size: int = 32
    lr: float = 1e-4
    alpha: float = 1.0
    gamma: float = 0.99
    num_epochs: int = 10
    num_workers: int = 12
    limit: int = 3000000
    pts_weight: list[float] = [10.0, 4.0, -4.0, -10.0]
    weight_decay: float = 0.0
    aux_weight: float = 0.0
    value_coef: float = 0.0
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.q_network.QNetwork"
    dataset_class: str = "riichienv_ml.datasets.mjai_logs.MCDataset"
    encoder_class: str = "riichienv_ml.features.feat_v1.ObservationEncoder"
    # Third-party evaluator
    evaluator: EvaluatorConfig = EvaluatorConfig()


class BcConfig(OfflineTrainConfig):
    # Online teacher BC (vs offline logs BC)
    online: bool = False
    offline_algorithm: Literal["cql", "behavior_cloning"] = "cql"
    load_model: str | None = None
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    # LR scheduler
    lr_min: float = 1e-5
    label_smoothing: float = 0.0
    shuffle_buffer_files: int = 1
    warmup_steps: int = 0  # linear warmup (in collection rounds, same unit as num_steps)
    max_grad_norm: float = 10.0  # gradient clipping max norm
    # Online teacher settings (used when online=True)
    teacher_model_name: Literal["kanachan", "mortal"] = "kanachan"  # model type ("kanachan", "mortal"...)
    teacher_model_path: str | None = None
    num_ray_workers: int = 4
    num_envs_per_worker: int = 16
    num_steps: int = 500
    train_epochs: int = 3  # epochs per collection round
    worker_device: Literal["cpu", "cuda"] = "cpu"
    gpu_per_worker: float = 0.0


class CqlConfig(OfflineTrainConfig):
    pass


class PpoConfig(WandbConfig):
    game: GameConfig = GameConfig()
    # Algorithm: "dqn" (DQN + CQL) or "ppo" (Actor-Critic + PPO)
    algorithm: Literal["dqn", "ppo"] = "ppo"
    load_model: str | None = None
    device: str = "cuda"
    num_workers: int = 12
    num_steps: int = 5000000
    batch_size: int = 128
    lr: float = 1e-4
    lr_min: float = 1e-6
    max_grad_norm: float = 1.0
    # DQN-specific params
    alpha_cql_init: float = 1.0
    alpha_cql_final: float = 0.1
    alpha_kl: float = 0.0
    alpha_kl_warmup_steps: int = 0
    # Exploration strategy (DQN only): "epsilon_greedy" or "boltzmann"
    exploration: Literal["epsilon_greedy", "boltzmann"] = "boltzmann"
    # epsilon-greedy params
    epsilon_start: float = 0.1
    epsilon_final: float = 0.01
    # Boltzmann (softmax) exploration params
    boltzmann_epsilon: float = 0.02
    boltzmann_temp_start: float = 0.1
    boltzmann_temp_final: float = 0.05
    top_p: float = 1.0
    capacity: int = 1000000
    # PPO-specific params
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    # Target network (for TD target computation)
    target_update_freq: int = 2000
    # Common params
    eval_interval: int = 2000
    eval_episodes: int = 100
    weight_sync_freq: int = 10
    worker_device: Literal["cpu", "cuda"] = "cpu"
    gpu_per_worker: float = 0.1
    num_envs_per_worker: int = 16
    gamma: float = 0.99
    weight_decay: float = 0.0
    aux_weight: float = 0.0
    entropy_coef_online: float = 0.0
    checkpoint_dir: str = "checkpoints"
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.actor_critic.ActorCriticNetwork"
    encoder_class: str = "riichienv_ml.features.feat_v1.ObservationEncoder"
    # Data collection
    collect_hero_only: bool = False
    # GRP reward shaping (per-kyoku reward)
    grp_model: str | None = None
    pts_weight: list[float] = [10.0, 4.0, -4.0, -10.0]
    async_rollout: bool = False
    # Self-play baseline update interval (0 = never update, keep frozen)
    baseline_update_interval: int = 0
    # Fixed opponent model path (if set, baseline is loaded from this path instead of hero weights)
    baseline_model: str | None = None
    # Third-party evaluator
    evaluator: EvaluatorConfig = EvaluatorConfig()


class HandPredConfig(WandbConfig):
    """Config for hand prediction (opponent hand estimation)."""

    game: GameConfig = GameConfig()
    data_glob: str = "/data/mjsoul/mjsoul-4p/2024/**/*.jsonl.gz"
    val_data_glob: str = "/data/mjsoul/mjsoul-4p/2024/01/**/*.jsonl.gz"
    output: str = "hand_pred_model.pth"
    device: str = "cuda"
    batch_size: int = 128
    num_workers: int = 12
    num_epochs: int = 10
    lr: float = 5e-4
    lr_eta_min: float = 1e-7
    weight_decay: float = 0.01
    max_grad_norm: float = 10.0
    samples_per_file: int = 128
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.hand_pred.HandPredCNN"
    encoder_class: str = "riichienv_ml.features.feat_v1.ObservationEncoder"
    dataset_class: str = "riichienv_ml.datasets.hand_pred.HandPredDataset"
    loss_type: str = "cross_entropy"  # "cross_entropy", "smooth_l1", or "mse"
    label_smoothing: float = 0.1
    sum_constraint_weight: float = 0.1  # auxiliary loss weight (regression only)
    val_step_interval: int = 20000  # run validation & save checkpoint every N steps


class BeliefStratifiedSampleKeepProbConfig(BaseModel):
    """Decision keep probabilities keyed by the most valuable target opponent."""

    enabled: bool = True
    riichi: float = 0.4
    meld1: float = 0.1
    meld2: float = 0.2
    meld3plus: float = 0.5
    closed_start: float = 0.01
    closed_end: float = 0.3
    closed_start_discard_count: int = 0
    closed_end_discard_count: int = 20


class BeliefSamplerConfig(WandbConfig):
    """Config for the joint hidden-allocation belief sampler."""

    game: GameConfig = GameConfig(n_players=4, replay_rule="tenhou")
    data_glob: str = "data/self_match/BC/test10/game_*.jsonl"
    val_data_glob: str = ""
    output: str = "models/belief_sampler/test01/model.pth"
    load_model: str | None = None
    device: str = "cuda"
    batch_size: int = 32
    lr: float = 1e-4
    lr_min: float = 1e-6
    num_epochs: int = 3
    num_workers: int = 8
    limit: int = 1000000
    weight_decay: float = 0.01
    max_grad_norm: float = 10.0
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    skip_single_action: bool = True
    shuffle_buffer_files: int = 1
    sample_keep_prob: float = 1.0
    stratified_sample_keep_prob: BeliefStratifiedSampleKeepProbConfig | None = None
    eval_num_samples: int = 4
    eval_sample_batches: int = 2
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.belief_allocation.JointHiddenAllocationSampler"
    dataset_class: str = "riichienv_ml.datasets.belief_allocation.BeliefAllocationDataset"
    encoder_class: str = "riichienv_ml.features.belief_features.BeliefFeatureEncoder"


class BeliefLogSamplingConfig(BaseModel):
    """Config for annotating MJAI logs with belief-sampler hidden hand samples."""

    game: GameConfig = GameConfig(n_players=4, replay_rule="tenhou")
    input_glob: str = "data/self_match/BC/test10/game_*.jsonl"
    output_dir: str = "data/belief_samples/test01"
    summary_path: str | None = None
    model_path: str = "models/belief_sampler/test01/model.pth"
    device: str = "cuda"
    num_logs: int = 10
    num_samples: int = 10
    batch_size: int = 32
    skip_single_action: bool = True
    overwrite: bool = False
    compress_output: bool = True
    seed: int | None = None
    temperature: float = 1.0
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    inference_dtype: Literal["fp32", "bf16"] = "bf16"
    metadata_key: str = "belief_allocation"
    response_metadata_key: str = "belief_response_allocations"
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.belief_allocation.JointHiddenAllocationSampler"
    encoder_class: str = "riichienv_ml.features.belief_features.BeliefFeatureEncoder"


class BeliefSamplingBenchmarkConfig(BaseModel):
    """Config for fixed-observation belief-sampler throughput benchmarking."""

    game: GameConfig = GameConfig(n_players=4, replay_rule="tenhou")
    input_glob: str = "data/self_match/BC/test10/game_*.jsonl"
    output_dir: str = "data/belief_sampling_benchmark/test01"
    summary_path: str | None = None
    decisions_csv_path: str | None = None
    prog_len_csv_path: str | None = None
    prog_len_bucket_csv_path: str | None = None
    model_path: str = "models/belief_sampler/test01/model.pth"
    device: str = "cuda"
    num_logs: int = 10
    max_decisions: int | None = None
    decision_stride: int = 1
    samples_per_call: int = 256
    target_duration_ms: float = 100.0
    warmup_calls: int = 1
    warmup_samples_per_call: int | None = None
    skip_single_action: bool = True
    overwrite: bool = False
    seed: int | None = None
    temperature: float = 1.0
    decode_steps: int | None = None
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    inference_dtype: Literal["fp32", "bf16"] = "bf16"
    progress_interval: int = 50
    include_decisions_in_summary: bool = True
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.belief_allocation.JointHiddenAllocationSampler"
    encoder_class: str = "riichienv_ml.features.belief_features.BeliefFeatureEncoder"


class BeliefAllocationEvaluationConfig(BaseModel):
    """Config for evaluating belief-sampler hidden-allocation quality."""

    game: GameConfig = GameConfig(n_players=4, replay_rule="tenhou")
    input_glob: str = "data/self_match/BC/test10/game_*.jsonl"
    output_dir: str = "data/belief_allocation_evaluation/test01"
    report_path: str | None = None
    summary_path: str | None = None
    decisions_csv_path: str | None = None
    opponents_csv_path: str | None = None
    opponent_state_csv_path: str | None = None
    opponent_discard_count_csv_path: str | None = None
    opponent_state_discard_count_csv_path: str | None = None
    shanten_distribution_csv_path: str | None = None
    model_path: str = "models/belief_sampler/test01/model.pth"
    device: str = "cuda"
    num_logs: int = 10
    max_decisions: int | None = None
    decision_stride: int = 1
    sample_keep_prob: float = 1.0
    samples_per_decision: int = 64
    batch_size: int = 4
    warmup_batches: int = 1
    skip_single_action: bool = True
    overwrite: bool = False
    seed: int | None = 0
    temperature: float = 1.0
    decode_steps: int | None = None
    matmul_precision: Literal["highest", "high", "medium"] = "high"
    inference_dtype: Literal["fp32", "bf16"] = "bf16"
    progress_interval: int = 50
    include_decisions_in_summary: bool = False
    include_opponents_in_summary: bool = False
    semantic_features: bool = True
    dora_weight: float = 1.0
    red_weight: float = 1.0
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.belief_allocation.JointHiddenAllocationSampler"
    encoder_class: str = "riichienv_ml.features.belief_features.BeliefFeatureEncoder"


class SelfMatchAgentConfig(BaseModel):
    config_path: str
    model_path: str
    device: str = "cuda"
    name: str | None = None


class SelfMatchConfig(BaseModel):
    game: GameConfig = GameConfig(n_players=4, replay_rule="tenhou")
    agents: list[SelfMatchAgentConfig] = []
    output_dir: str = "data/self_match/BC/test01"
    summary_path: str | None = None
    num_games: int = 1000
    base_seed: int = 0
    seed_stride: int = 1
    progress_interval: int = 10
    overwrite: bool = False
    compress_logs: bool = False
    skip_mjai_logging: bool = False
    validate_saved_logs: bool = False
    log_policy_meta: bool = False

    @model_validator(mode="after")
    def validate_agents(self) -> SelfMatchConfig:
        if not self.agents:
            raise ValueError("self_match.agents must contain at least one agent config")
        if len(self.agents) not in (1, self.game.n_players):
            raise ValueError(
                f"self_match.agents must contain either 1 or {self.game.n_players} entries (got {len(self.agents)})"
            )
        if self.num_games <= 0:
            raise ValueError("self_match.num_games must be > 0")
        if self.progress_interval <= 0:
            raise ValueError("self_match.progress_interval must be > 0")
        if self.log_policy_meta and self.skip_mjai_logging:
            raise ValueError("self_match.log_policy_meta requires skip_mjai_logging=false")
        return self


class Config(BaseModel):
    grp: GrpConfig = GrpConfig()
    bc: BcConfig = BcConfig()
    cql: CqlConfig = CqlConfig()
    ppo: PpoConfig = PpoConfig()
    hand_pred: HandPredConfig = HandPredConfig()
    belief_sampler: BeliefSamplerConfig = BeliefSamplerConfig()
    belief_log_sampling: BeliefLogSamplingConfig = BeliefLogSamplingConfig()
    belief_sampling_benchmark: BeliefSamplingBenchmarkConfig = BeliefSamplingBenchmarkConfig()
    belief_allocation_evaluation: BeliefAllocationEvaluationConfig = BeliefAllocationEvaluationConfig()
    self_match: SelfMatchConfig = SelfMatchConfig(
        agents=[
            SelfMatchAgentConfig(
                config_path="riichienv-ml/src/riichienv_ml/configs/4p/bc_tenhou_seq_test01.yml",
                model_path="models/behavior_cloning/test01/model.pth",
            )
        ]
    )


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
