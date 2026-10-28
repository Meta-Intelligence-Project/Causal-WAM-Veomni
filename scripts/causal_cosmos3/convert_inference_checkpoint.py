"""Convert legacy Cosmos DCP into complete native/HF weights before inference."""

import argparse
import hashlib
import json
import math
import pathlib
import pickle

import torch
import yaml
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import StorageMeta

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models import build_foundation_model
from veomni.models.diffusers.causal_cosmos3.cosmos3_transformer.configuration_cosmos3_transformer import Cosmos3Config


def cosmos_parameter_name(name):
    """Map dense Cosmos MoT tensors without transposing their linear weights."""
    if not name.startswith("net."):
        raise ValueError(f"Unsupported Cosmos checkpoint key: {name}")
    name = (
        name.removeprefix("net.").replace("language_model.model.", "").replace("language_model.lm_head.", "lm_head.")
    )
    prefixes = {
        "vae2llm.": "proj_in.",
        "llm2vae.": "proj_out.",
        "action2llm.": "action_proj_in.",
        "llm2action.": "action_proj_out.",
        "state2llm.": "state_proj_in.",
        "time_embedder.mlp.0.": "time_embedder.linear_1.",
        "time_embedder.mlp.2.": "time_embedder.linear_2.",
    }
    for old, new in prefixes.items():
        if name.startswith(old):
            name = new + name[len(old) :]
            break
    replacements = {
        "q_proj": "to_q",
        "k_proj": "to_k",
        "v_proj": "to_v",
        "o_proj": "to_out",
        "q_proj_moe_gen": "add_q_proj",
        "k_proj_moe_gen": "add_k_proj",
        "v_proj_moe_gen": "add_v_proj",
        "o_proj_moe_gen": "to_add_out",
        "q_norm_moe_gen": "norm_added_q",
        "k_norm_moe_gen": "norm_added_k",
    }
    parts = name.split(".")
    if "self_attn" in parts:
        i = parts.index("self_attn") + 1
        parts[i] = replacements.get(parts[i], parts[i])
    return ".".join(parts)


class _PathCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Python 3.13 records these paths differently from the supported 3.12 runtime.
        if module == "pathlib._local" and name in ("Path", "PosixPath", "PurePath", "PurePosixPath"):
            return getattr(pathlib, name)
        return super().find_class(module, name)


class CosmosCheckpointReader(FileSystemReader):
    """Read an explicitly trusted local Cosmos DCP without changing global pickle."""

    def read_metadata(self, *args, **kwargs):
        with self.fs.create_stream(self._get_metadata_path(kwargs.get("rank")), "rb") as stream:
            metadata = _PathCompatUnpickler(stream).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        return metadata


def convert_cosmos_checkpoint(
    checkpoint, output, *, architecture, training_config=None, dtype=torch.bfloat16, shard_bytes=512 * 1024**2
):
    """Stream only model tensors from Cosmos DCP; never read optimizer/job RNG."""
    import torch.distributed.checkpoint as dcp

    checkpoint, output = pathlib.Path(checkpoint).resolve(), pathlib.Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Conversion destination already exists: {output}")
    training_config = pathlib.Path(training_config) if training_config else checkpoint.parent.parent / "config.yaml"
    trained = yaml.load(training_config.read_text(), Loader=yaml.BaseLoader)
    source_config = trained["model"]["config"]
    if source_config.get("action_gen") != "true" or source_config.get("sound_gen") == "true":
        raise ValueError("Conversion supports a trained causal action model without audio")
    reader = CosmosCheckpointReader(checkpoint / "model")
    metadata = reader.read_metadata()
    tensors = {k: v for k, v in metadata.state_dict_metadata.items() if hasattr(v, "size")}
    mapping = {name: cosmos_parameter_name(name) for name in tensors}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Duplicate target tensors in Cosmos checkpoint")
    config = Cosmos3Config.from_json_file(str(architecture))
    config.action_interface = "shared"
    config.action_dim = int(source_config["max_action_dim"])
    config.state_dim = tensors["net.state2llm.weight"].size[1]
    config.qk_norm_for_text = source_config["vlm_config"]["model_instance"]["config"]["qk_norm_for_text"] == "true"
    config.use_und_k_norm_for_gen = (
        source_config["vlm_config"]["model_instance"]["config"]["use_und_k_norm_for_gen"] == "true"
    )
    config.timestep_scale = float(source_config["diffusion_expert_config"]["timestep_range"]) / 1000
    if isinstance(fixed_step := source_config.get("fixed_step_sampler_config"), dict):
        config.fixed_step_sampler_config = dict(
            t_list=[float(sigma) for sigma in fixed_step["t_list"]], sample_type=fixed_step.get("sample_type", "sde")
        )
    model = build_foundation_model(
        config,
        torch_dtype=str(dtype).split(".")[-1],
        init_device="meta",
        ops_implementation=OpsImplementationConfig(
            attn_implementation="eager", rotary_pos_emb_implementation="eager", rms_norm_implementation="eager"
        ),
    )
    expected = {key: tuple(value.shape) for key, value in model.named_parameters()}
    missing = sorted(set(expected) - set(mapping.values()))
    unexpected = sorted(set(mapping.values()) - set(expected))
    mismatched = {
        src: dict(source=list(tensors[src].size), target=list(expected[dst]))
        for src, dst in mapping.items()
        if dst in expected and tuple(tensors[src].size) != expected[dst]
    }
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Incompatible Cosmos checkpoint: missing={missing}, unexpected={unexpected}, shapes={mismatched}"
        )
    output.mkdir(parents=True)
    config.save_pretrained(output)
    report = dict(
        source_checkpoint=str(checkpoint),
        training_config=str(training_config),
        source_metadata_sha256=hashlib.sha256((checkpoint / "model/.metadata").read_bytes()).hexdigest(),
        source_training_config_sha256=hashlib.sha256(training_config.read_bytes()).hexdigest(),
        missing=missing,
        unexpected=unexpected,
        mismatched=mismatched,
        keys={},
        dtype=str(dtype),
        parameter_layout="linear matrices unchanged; explicit name mapping only",
    )
    weight_map, total_size, group, group_bytes, part = {}, 0, [], 0, 0

    def export(names, number):
        state = {
            name: torch.empty(tuple(tensors[name].size), dtype=tensors[name].properties.dtype, device="cpu")
            for name in names
        }
        dcp.load_state_dict(
            state, storage_reader=reader, no_dist=True, planner=dcp.DefaultLoadPlanner(allow_partial_load=True)
        )
        converted = {mapping[name]: value.to(dtype=dtype).contiguous() for name, value in state.items()}
        filename = f"model-{number:05d}.safetensors"
        save_file(converted, output / filename, metadata={"format": "pt"})
        for name, value in state.items():
            destination = mapping[name]
            weight_map[destination] = filename
            report["keys"][name] = dict(
                target=destination,
                shape=list(value.shape),
                source_dtype=str(value.dtype),
                target_dtype=str(dtype),
                conversion_max_abs=float((converted[destination].float() - value.float()).abs().max()),
            )
        print(f"Converted shard {number}: {len(names)} tensors", flush=True)

    for name, value in tensors.items():
        count = math.prod(value.size) * torch.empty((), dtype=value.properties.dtype).element_size()
        if group and group_bytes + count > shard_bytes:
            part += 1
            export(group, part)
            group, group_bytes = [], 0
        group.append(name)
        group_bytes += count
        total_size += math.prod(value.size) * torch.empty((), dtype=dtype).element_size()
    if group:
        export(group, part + 1)
    (output / "model.safetensors.index.json").write_text(
        json.dumps(dict(metadata=dict(total_size=total_size), weight_map=weight_map), indent=2) + "\n"
    )
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-config")
    parser.add_argument("--model-config", default="configs/model_configs/causal_cosmos3/causal_cosmos3_edge.json")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    report = convert_cosmos_checkpoint(
        args.checkpoint,
        args.output,
        architecture=args.model_config,
        training_config=args.training_config,
        dtype=getattr(torch, args.dtype),
    )
    print(f"Complete trained checkpoint: {len(report['keys'])} tensors -> {args.output}")


if __name__ == "__main__":
    main()
