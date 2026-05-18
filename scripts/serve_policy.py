import sys
import os
_openpi_src = "/home/chengyuxuan/openpi/src"
if _openpi_src not in sys.path:
    sys.path.insert(0, _openpi_src)
import dataclasses
import enum
import logging
import socket
import time
import numpy as np
import tyro
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
os.environ["OPENPI_DATA_HOME"] = "/share/chengyuxuan-local/openpi" #所有的文件不要乱放 全部放在这个/share文件夹下面
os.environ.setdefault("OPENPI_DUQUANT_PACKDIR", "/home/chengyuxuan/openpi/src/openpi/models_pytorch/quant/duquant_packed")  # SVD 分解缓存目录
os.environ.setdefault("OPENPI_DUQUANT_INCLUDE", r"paligemma_with_expert\.paligemma\.model\..*")
os.environ.setdefault("OPENPI_DUQUANT_EXCLUDE", "")

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(
        default_factory=lambda: Checkpoint(
            config="pi05_libero",
            dir="/share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch",
        )
    )

    # Whether to quantize the model with DuQuant.
    quantize: bool = False


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None, quantize: bool = False) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt, quantize=quantize
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt, quantize=args.quantize
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, quantize=args.quantize)


def warmup_policy(policy: _policy.Policy, num_warmup: int = 3) -> None:
    """Run a few warmup inferences to JIT-compile and fill GPU kernels before serving.

    Args:
        policy: The policy to warm up.
        num_warmup: Number of warmup iterations to run.
    """
    logging.info(f"Warming up policy ({num_warmup} iterations)...")
    start = time.monotonic()

    fake_obs = {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/image": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "warmup",
    }

    for i in range(num_warmup):
        policy.infer(fake_obs)
        logging.info(f"  warmup {i + 1}/{num_warmup} done")

    elapsed = time.monotonic() - start
    logging.info(f"Warmup complete in {elapsed:.1f}s")


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Warm up the policy (JIT compilation + GPU kernel filling).
    # warmup_policy(policy)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    # ── 量化模式预设（修改这里切换模式）──
    # 有效值: 0=FP16 | 1=W4A4 | 2=W4A8 | 3=W2A2 | 4=DRYRUN
    # 也可以通过环境变量 OPENPI_QUANT_MODE 传入，例如:
    #   OPENPI_QUANT_MODE=2 python scripts/serve_policy.py ...
    _QUANT_MODES = {
        "0": {},                                  # FP16
        "1": {"OPENPI_DUQUANT_DRYRUN": "0", "OPENPI_DUQUANT_WBITS_DEFAULT": "4", "OPENPI_DUQUANT_ABITS": "4"},  # W4A4
        "2": {"OPENPI_DUQUANT_DRYRUN": "0", "OPENPI_DUQUANT_WBITS_DEFAULT": "4", "OPENPI_DUQUANT_ABITS": "8"},  # W4A8
        "3": {"OPENPI_DUQUANT_DRYRUN": "0", "OPENPI_DUQUANT_WBITS_DEFAULT": "4", "OPENPI_DUQUANT_ABITS": "16"},  # W2A2
        "4": {"OPENPI_DUQUANT_DRYRUN": "0", "OPENPI_DUQUANT_WBITS_DEFAULT": "4", "OPENPI_DUQUANT_ABITS": "8"},  # DRYRUN
    }
    quant_mode = os.environ.get("OPENPI_QUANT_MODE", "0")
    for k, v in _QUANT_MODES.get(quant_mode, {}).items():
        os.environ[k] = v
    if quant_mode != "0":
        logging.info("[Quant] Mode=%s  WBITS=%s  ABITS=%s  DRYRUN=%s",
                     quant_mode,
                     os.environ.get("OPENPI_DUQUANT_WBITS_DEFAULT", "-"),
                     os.environ.get("OPENPI_DUQUANT_ABITS", "-"),
                     os.environ.get("OPENPI_DUQUANT_DRYRUN", "0"))

    main(tyro.cli(Args))

'''
如果需要指定参数 环境变量指定具体的东西替换   OPENPI_QUANT_MODE=2 
CUDA_VISIBLE_DEVICES=0   python scripts/serve_policy.py     --port 8000     --quantize     policy:checkpoint     --policy.config pi05_libero     --policy.dir /share/chengyuxuan-local/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
'''
