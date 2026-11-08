"""Fixed-duration revision trainers using the corrected experiment's math.

Callers own provenance verification and supply only train/validation tensors.
This module never opens source datasets or holdout assets.  Full and No-FU use
the same SAFER sampler and RNG consumption; No-FU never receives FU examples.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Callable

import numpy as np
import torch
from torch import nn

from fall_pipeline.joint import reconstruction_core as core
from fall_pipeline.safer.train_f0b_safer_heads import classification_metrics, selection_key


class TrainingPaused(RuntimeError):
    """A pause request was observed before starting another complete epoch."""


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path, writer, binary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    _atomic(path, lambda stream: json.dump(value, stream, ensure_ascii=False,
                                          indent=2, allow_nan=False), False)


def save_tensor(path, value):
    _atomic(path, lambda stream: torch.save(value, stream), True)


def _save_array(path, value):
    _atomic(path, lambda stream: np.save(stream, value, allow_pickle=False), True)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path, contract=None):
    result = torch.load(path, map_location="cpu", weights_only=True)
    if contract is not None:
        require(result["contract"] == contract, "Checkpoint contract changed: " + str(path))
    return result


def _pause(output):
    if output is not None and (Path(output) / "PAUSE_REQUESTED").exists():
        raise TrainingPaused("Pause requested at completed epoch boundary")


def _guard(folder, contract):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_contract.json"
    if path.exists():
        require(json.loads(path.read_text()) == contract, "Run contract changed: " + str(folder))
    else:
        require(not any(folder.iterdir()), "Refusing nonempty uncontracted training directory: " + str(folder))
        atomic_json(path, contract)


def _contract(contract, protocol, **extra):
    # JSON normalization rejects unserializable provenance before any training.
    value = {**copy.deepcopy(contract), "trainer": "revision-fixed-v1",
             "protocol": copy.deepcopy(protocol), "trainer_sha256": sha256(__file__), **extra}
    return json.loads(json.dumps(value, allow_nan=False))


def _completed(folder, contract):
    path = folder / "selection.json"
    if not path.exists():
        return None
    selected = json.loads(path.read_text())
    require(selected["completed"] and selected["contract"] == contract,
            "Completed stage contract changed")
    require(sha256(folder / "best.pt") == selected["checkpoint_sha256"],
            "Selected checkpoint changed")
    if "selected_validation_logits_sha256" in selected:
        require(sha256(folder / "selected_validation_logits.npy") ==
                selected["selected_validation_logits_sha256"], "Selected logits changed")
    return load_checkpoint(folder / "best.pt", contract)


def _arrays(data, feature_dim, device):
    x, y = data["x"], data["y"]
    require(x.ndim == 2 and x.shape == (len(y), feature_dim) and len(y) > 0,
            "Invalid feature dimensions")
    require(x.dtype == torch.float32 and y.dtype == torch.int64 and y.ndim == 1,
            "Features must be float32; labels must be int64")
    require(x.device == y.device and x.device == torch.device(device), "Tensor device mismatch")
    require(all(bool(torch.isfinite(part).all()) for part in x.split(8192))
            and bool(((y >= 0) & (y < 4)).all()),
            "Invalid source features or labels")


def consume_fu_dropout(model, config):
    """Consume exactly the FU adapter dropout RNG without FU features/labels.

    LayerNorm/Linear/GELU/head/CE/backward have no stochastic operations.  FU
    batches are always full, including SAFER's last batch; only this dropout
    consumes global RNG.  Use its actual module/device/dtype and bottleneck
    shape, with no graph and no optimizer contribution.
    """
    dropout = model.adapter.dropout
    parameter = model.adapter.down.weight
    with torch.no_grad():
        dummy = torch.empty((config["training"]["fu_batch_size"],
                             config["model"]["bottleneck"]),
                            device=parameter.device, dtype=parameter.dtype)
        # A finite constant avoids propagating uninitialized values in checks.
        dummy.fill_(1)
        dropout(dummy)


def train_joint_epoch(model, optimizer, safer, fu, epoch, stage, protocol,
                      condition="full", before_safer: Callable | None = None):
    """Corrected train_epoch, with an explicit FU-loss removal condition."""
    require(condition in ("full", "no_fu"), "Unknown joint condition")
    require((fu is None) == (condition == "no_fu"),
            "Full requires FU tensors; No-FU must not receive them")
    p = protocol["training"]
    factor = p["gamma"] ** sum(epoch - 1 >= milestone for milestone in p[stage + "_schedule"])
    bases = [p["j0_lr"]] if stage == "j0" else [p["adapter_lr"], p["head_lr"]]
    for group, base in zip(optimizer.param_groups, bases):
        group["lr"] = base * factor
    model.train()
    sy = safer["y"]
    sw = core.weights(sy, 4)
    if condition == "full":
        fu_train = np.arange(len(fu["y"]))
        fw = core.weights(fu["y"][fu_train], 2)
    else:
        # Placeholder sampler indices contain no FU metadata.  Its independent
        # CPU generator cannot change global RNG or the SAFER order.
        fu_train = np.arange(1)
    total = np.zeros(2)
    steps = 0
    for si, fi in core.paired_batches(len(sy), fu_train, epoch, protocol):
        si = si.to(sy.device)
        optimizer.zero_grad(set_to_none=True)
        if before_safer is not None:
            before_safer(epoch, steps, rng_state())
        sl = nn.functional.cross_entropy(model(safer["x"][si], "safer"), sy[si], weight=sw)
        if condition == "full":
            fi = fi.to(sy.device)
            fl = nn.functional.cross_entropy(model(fu["x"][fi], "fu"), fu["y"][fi], weight=fw)
            loss = sl + fl
        else:
            if stage == "j1":
                consume_fu_dropout(model, protocol)
            fl = None
            loss = sl  # coefficient is one; never divide by the number of tasks
        require(bool(torch.isfinite(loss)), "Nonfinite joint loss; epoch not committed")
        loss.backward()
        optimizer.step()
        total += np.array([sl.item(), fl.item() if fl is not None else 0.])
        steps += 1
    require(steps > 0, "No joint training steps")
    return {"safer": float(total[0] / steps), "fu": float(total[1] / steps), "steps": steps}


def run_joint_stage(folder, stage, protocol, contract, device, train, val, fu=None,
                    *, condition="full", initial=None, fixed_epochs=None,
                    output=None, stop_after=None):
    """Fixed J0/J1 duration, atomic epoch resume, and final-epoch selection.

    `stop_after` is an absolute completed epoch, used only in synthetic tests.
    Production calls pass the previously selected 21/12 budgets explicitly.
    """
    require(stage in ("j0", "j1"), "Unknown joint stage")
    require(condition in ("full", "no_fu"), "Unknown joint condition")
    require((fu is None) == (condition == "no_fu"), "Invalid FU access for condition")
    require(fixed_epochs is not None and int(fixed_epochs) == fixed_epochs and fixed_epochs > 0,
            "Explicit positive fixed duration required")
    _arrays(train, protocol["model"]["feature_dim"], device)
    _arrays(val, protocol["model"]["feature_dim"], device)
    folder = Path(folder)
    run_contract = _contract(contract, protocol, stage=stage, condition=condition,
                             run_mode="fixed_duration_refit", epoch_budget=fixed_epochs,
                             initial_state_sha256=state_digest(initial) if initial is not None else None)
    _guard(folder, run_contract)
    done = _completed(folder, run_contract)
    if done is not None:
        return done
    model = core.model_for(stage, protocol, device, initial)
    optimizer = core.optimizer_for(model, stage, protocol)
    latest = folder / "latest.pt"
    history, start = [], 1
    if latest.exists():
        saved = load_checkpoint(latest, run_contract)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, start = saved["history"], saved["epoch"] + 1
        restore_rng(saved["rng"])
    elif stage == "j1":
        # Preserve the corrected runner's exact order: constructing reference
        # resets CPU/CUDA seeds after constructing J1 and before dropout.
        reference = core.model_for("j0", protocol, device)
        reference.load_state_dict(initial)
        if condition == "full":
            audit = core.epoch0_exact(reference, model, train, fu, np.arange(len(fu["y"])), protocol)
        else:
            batch = protocol["training"]["eval_batch_size"]
            a = core.predict(reference, train["x"], "safer", batch)
            b = core.predict(model, train["x"], "safer", batch)
            require(np.array_equal(a, b), "J1 epoch zero is not exact J0")
            audit = {"passed": True}
        audit["scope"] = "all SAFER training inputs" + (" and FU training clips" if condition == "full" else "; FU inputs never accessed")
        atomic_json(folder / "epoch0_audit.json", audit)
        del reference
    for epoch in range(start, fixed_epochs + 1):
        _pause(output)
        loss = train_joint_epoch(model, optimizer, train, fu, epoch, stage, protocol, condition)
        row = {"epoch": epoch, "loss": loss}
        history.append(row)
        save_tensor(latest, {"contract": run_contract, "model": cpu_state(model),
                             "optimizer": optimizer.state_dict(), "history": history,
                             "epoch": epoch, "rng": rng_state()})
        atomic_json(folder / "history.json", history)
        print(f"{condition}/{stage} epoch {epoch}/{fixed_epochs}", flush=True)
        if stop_after is not None and epoch >= stop_after:
            return None
    selected = {"completed": True, "model": cpu_state(model), "contract": run_contract,
                "epoch": fixed_epochs, "fixed_duration": True}
    save_tensor(folder / "best.pt", selected)
    atomic_json(folder / "selection.json", {key: value for key, value in selected.items() if key != "model"} |
                {"checkpoint_sha256": sha256(folder / "best.pt"), "holdout_used_for_selection": False})
    return selected


def new_classifier(feature_dim, seed, device):
    layer = nn.Linear(feature_dim, 4)
    # Preserve the original reset AFTER Linear's default initialization.
    torch.manual_seed(seed)
    nn.init.normal_(layer.weight, 0, .01)
    nn.init.zeros_(layer.bias)
    return layer.to(device)


def adapt_copy(data, adapter, batch=1024):
    """Return an independent adapted feature tensor; never mutate raw caches."""
    result = {**data, "x": data["x"].clone(), "y": data["y"]}
    adapter.eval()
    with torch.inference_mode():
        for start in range(0, len(result["x"]), batch):
            result["x"][start:start + batch] = adapter(result["x"][start:start + batch])
            require(bool(torch.isfinite(result["x"][start:start + batch]).all()),
                    "Nonfinite adapted features")
    return result


def classifier_logits(head, data, batch):
    head.eval()
    with torch.inference_mode():
        return np.concatenate([head(data["x"][i:i + batch]).cpu().numpy()
                               for i in range(0, len(data["y"]), batch)])


def fit_classifier_arrays(folder, protocol, contract, train, val, device,
                          output=None, stop_after=None):
    """Corrected weighted CE/SGD/50-epoch recipe with validation-only selection."""
    folder = Path(folder)
    p = protocol["training"]
    _arrays(train, train["x"].shape[1], device)
    _arrays(val, train["x"].shape[1], device)
    require(p["epochs"] > 0, "Positive classifier epoch budget required")
    run_contract = _contract(contract, protocol, stage="classifier", epoch_budget=p["epochs"])
    _guard(folder, run_contract)
    done = _completed(folder, run_contract)
    if done is not None:
        return done
    head = new_classifier(train["x"].shape[1], p["seed"], device)
    optimizer = torch.optim.SGD(head.parameters(), lr=p["lr"], momentum=p["momentum"],
                                weight_decay=p["weight_decay"])
    latest = folder / "latest.pt"
    history, best, best_model, start = [], None, None, 1
    if latest.exists():
        saved = load_checkpoint(latest, run_contract)
        head.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_model, start = saved["history"], saved["best"], saved["best_model"], saved["epoch"] + 1
        restore_rng(saved["rng"])
    class_weights = core.weights(train["y"], 4)
    for epoch in range(start, p["epochs"] + 1):
        _pause(output)
        head.train()
        order = torch.randperm(len(train["y"]), generator=torch.Generator().manual_seed(p["seed"] + epoch))
        total, steps = 0., 0
        for start_i in range(0, len(order), p["batch_size"]):
            indices = order[start_i:start_i + p["batch_size"]].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(train["x"][indices]), train["y"][indices], weight=class_weights)
            require(bool(torch.isfinite(loss)), "Nonfinite classifier loss; epoch not committed")
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps += 1
        logits = classifier_logits(head, val, p["eval_batch_size"])
        metrics = classification_metrics(val["y"].cpu().numpy(), logits)
        row = {"epoch": epoch, "loss": total / steps, "metrics": metrics}
        history.append(row)
        # Strict > deliberately keeps the earliest exact tie.
        if best is None or selection_key(metrics) > selection_key(best["metrics"]):
            best, best_model = row, cpu_state(head)
        save_tensor(latest, {"contract": run_contract, "model": cpu_state(head),
                             "optimizer": optimizer.state_dict(), "epoch": epoch,
                             "history": history, "best": best, "best_model": best_model,
                             "rng": rng_state()})
        atomic_json(folder / "history.json", history)
        print(f"classifier epoch {epoch}/{p['epochs']}", flush=True)
        if stop_after is not None and epoch >= stop_after:
            return None
    head.load_state_dict(best_model)
    logits = classifier_logits(head, val, p["eval_batch_size"])
    require(classification_metrics(val["y"].cpu().numpy(), logits) == best["metrics"],
            "Selected classifier metrics do not reproduce")
    _save_array(folder / "selected_validation_logits.npy", logits)
    selected = {"completed": True, "model": best_model, "contract": run_contract, **best,
                "selected_validation_logits_sha256": sha256(folder / "selected_validation_logits.npy")}
    save_tensor(folder / "best.pt", selected)
    atomic_json(folder / "selection.json", {
        "completed": True, "contract": run_contract, "epoch": best["epoch"],
        "validation_metrics": best["metrics"], "checkpoint_sha256": sha256(folder / "best.pt"),
        "selected_validation_logits_sha256": selected["selected_validation_logits_sha256"],
        "rule": "validation macro-F1, conditional fall-versus-lying AP, earliest exact tie",
        "holdout_used_for_selection": False})
    return selected


def synthetic_fixture(device="cpu"):
    """Small, artificial tensors only; no paper examples or held-out data."""
    generator = torch.Generator().manual_seed(9081)
    def data(count):
        return {"x": torch.randn(count, 12, generator=generator).to(device),
                "y": (torch.arange(count) % 4).to(device)}
    joint = {
        "model": {"feature_dim": 12, "bottleneck": 256, "dropout": .1},
        "training": {"seed": 2, "j0_lr": .001, "head_lr": .00005,
                     "adapter_lr": .0001, "betas": [.9, .999], "eps": 1e-8,
                     "weight_decay": .0001, "gamma": .1,
                     "j0_schedule": [20], "j1_schedule": [12],
                     "safer_batch_size": 9, "fu_batch_size": 32, "eval_batch_size": 7}}
    classifier = {"training": {"seed": 2, "lr": .006, "momentum": .9,
                                "weight_decay": 0., "batch_size": 5,
                                "eval_batch_size": 7, "epochs": 3}}
    fu = {"x": torch.randn(11, 12, generator=generator).to(device),
          "y": (torch.arange(11) % 2).to(device)}
    return joint, classifier, data(23), data(12), fu


def _assert_same(a, b, prefix="state"):
    if isinstance(a, torch.Tensor):
        require(torch.equal(a, b), "Synthetic verification mismatch: " + prefix)
    elif isinstance(a, dict):
        require(a.keys() == b.keys(), "Synthetic verification keys: " + prefix)
        for key in a:
            _assert_same(a[key], b[key], prefix + "." + str(key))
    elif isinstance(a, (list, tuple)):
        require(len(a) == len(b), "Synthetic verification length: " + prefix)
        for i, (left, right) in enumerate(zip(a, b)):
            _assert_same(left, right, prefix + "." + str(i))
    else:
        require(a == b, "Synthetic verification mismatch: " + prefix)


def verify_training_contract(device="cpu"):
    """Synthetic acceptance checks, suitable for CPU or visible CUDA GPU zero.

    The 32 x 256 FU dropout shape is the production shape.  These tiny runs
    check implementation contracts and must not be reported as fall-detection
    experiments.  No dataset paths are read and all files are temporary.
    """
    device = torch.device(device)
    if device.type == "cuda":
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and device.index in (None, 0),
                "Synthetic CUDA verification requires physical GPU zero only")
        require(torch.cuda.is_available(), "CUDA is unavailable")
        device = torch.device("cuda:0")
    saved_rng = rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    old_threads = torch.get_num_threads()
    checks = []
    try:
        torch.use_deterministic_algorithms(True)
        if device.type == "cpu":
            torch.set_num_threads(1)
        protocol, classifier, train, val, fu = synthetic_fixture(device)
        raw_digest = state_digest({"train": train["x"], "val": val["x"]})
        initial = cpu_state(core.model_for("j0", protocol, device))
        for stage in ("j0", "j1"):
            first = core.model_for(stage, protocol, device, initial if stage == "j1" else None)
            opt = core.optimizer_for(first, stage, protocol)
            before = rng_state()
            expected = core.train_epoch(first, opt, train, fu, np.arange(len(fu["y"])), 1, stage, protocol)
            expected_state, expected_rng = cpu_state(first), rng_state()
            second = core.model_for(stage, protocol, device, initial if stage == "j1" else None)
            second_opt = core.optimizer_for(second, stage, protocol)
            restore_rng(before)
            actual = train_joint_epoch(second, second_opt, train, fu, 1, stage, protocol)
            _assert_same(expected, actual)
            _assert_same(expected_state, cpu_state(second))
            _assert_same(opt.state_dict(), second_opt.state_dict())
            _assert_same(expected_rng, rng_state())
        checks.append("full_j0_j1_exact_historical_core_math")

        other = copy.deepcopy(protocol)
        other["training"]["seed"] += 1
        require(state_digest(initial) != state_digest(cpu_state(core.model_for("j0", other, device))),
                "Different seeds did not change initialization")
        order_a = next(core.paired_batches(23, np.arange(11), 1, protocol))[0]
        order_b = next(core.paired_batches(23, np.arange(11), 1, other))[0]
        require(not torch.equal(order_a, order_b), "Different seeds did not change sampler")
        checks.append("different_seed_changes_initialization_and_order")

        with tempfile.TemporaryDirectory(prefix="revision-training-verify-") as temporary:
            root = Path(temporary)
            contract = {"scope": "synthetic verification only"}
            j0_states = {}
            for condition in ("full", "no_fu"):
                result = run_joint_stage(root / condition / "j0", "j0", protocol, contract,
                                         device, train, val, fu if condition == "full" else None,
                                         condition=condition, fixed_epochs=3)
                j0_states[condition] = result["model"]
            for key in j0_states["full"]:
                if key.startswith("safer_head."):
                    _assert_same(j0_states["full"][key], j0_states["no_fu"][key])
            checks.append("j0_safer_head_full_nofu_exact")

            traces = {}
            for condition in ("full", "no_fu"):
                model = core.model_for("j1", protocol, device, j0_states[condition])
                optimizer = core.optimizer_for(model, "j1", protocol)
                reference = core.model_for("j0", protocol, device)
                reference.load_state_dict(j0_states[condition])
                require(np.array_equal(core.predict(reference, train["x"], "safer", 7),
                                       core.predict(model, train["x"], "safer", 7)),
                        "J1 epoch zero changed SAFER logits")
                trace = []
                for epoch in range(1, 3):
                    train_joint_epoch(model, optimizer, train, fu if condition == "full" else None,
                                      epoch, "j1", protocol, condition,
                                      before_safer=lambda epoch, step, state: trace.append(
                                          {"cpu": state["cpu"], "cuda": state["cuda"]}))
                traces[condition] = trace
            _assert_same(traces["full"], traces["no_fu"])
            checks.extend(["j1_epoch_zero_identity_and_reference_reset",
                           "nofu_fu_dropout_rng_matches_before_every_safer_step"])

            for condition in ("full", "no_fu"):
                kwargs = dict(stage="j1", protocol=protocol, contract=contract, device=device,
                              train=train, val=val, fu=fu if condition == "full" else None,
                              condition=condition, initial=j0_states[condition], fixed_epochs=3)
                complete = root / condition / "j1_uninterrupted"
                interrupted = root / condition / "j1_resumed"
                expected = run_joint_stage(complete, **kwargs)
                require(run_joint_stage(interrupted, stop_after=1, **kwargs) is None,
                        "Synthetic interruption did not stop")
                actual = run_joint_stage(interrupted, **kwargs)
                _assert_same(expected, actual)
                _assert_same(load_checkpoint(complete / "latest.pt"), load_checkpoint(interrupted / "latest.pt"))
                _assert_same(actual, run_joint_stage(interrupted, **kwargs))
                try:
                    run_joint_stage(interrupted, **{**kwargs, "contract": {"scope": "changed"}})
                except RuntimeError as error:
                    require("contract changed" in str(error).lower(), "Wrong contract rejection")
                else:
                    raise RuntimeError("Changed joint contract was accepted")
            checks.extend(["full_nofu_j1_epoch_boundary_resume_exact",
                           "completed_run_idempotent_and_changed_contract_rejected"])

            kwargs = dict(protocol=classifier, contract=contract, train=train, val=val, device=device)
            expected = fit_classifier_arrays(root / "classifier_full", **kwargs)
            require(fit_classifier_arrays(root / "classifier_resume", stop_after=1, **kwargs) is None,
                    "Classifier interruption did not stop")
            actual = fit_classifier_arrays(root / "classifier_resume", **kwargs)
            _assert_same(expected, actual)
            _assert_same(load_checkpoint(root / "classifier_full/latest.pt"),
                         load_checkpoint(root / "classifier_resume/latest.pt"))
            checks.append("classifier_validation_selection_and_resume_exact")

            model = core.model_for("j1", protocol, device, initial)
            with torch.no_grad():
                model.adapter.up.bias.fill_(.2)
            adapted = adapt_copy(train, model.adapter, batch=7)
            require(adapted["x"].data_ptr() != train["x"].data_ptr() and
                    not torch.equal(adapted["x"], train["x"]), "Adapted tensor is not independent")
            require(raw_digest == state_digest({"train": train["x"], "val": val["x"]}),
                    "Synthetic raw features were modified")
            checks.append("raw_feature_tensors_unchanged")

            pause_root = root / "pause"
            pause_root.mkdir()
            (pause_root / "PAUSE_REQUESTED").touch()
            try:
                fit_classifier_arrays(pause_root / "head", output=pause_root, **kwargs)
            except TrainingPaused:
                require(not (pause_root / "head/latest.pt").exists(), "Paused epoch unexpectedly committed")
            else:
                raise RuntimeError("Pause request was ignored")
            checks.append("pause_before_next_epoch")
        return {"status": "PASS", "scope": "synthetic training-contract checks only",
                "device": str(device), "torch": str(torch.__version__),
                "fu_dropout_shape": [32, 256], "checks": checks}
    finally:
        restore_rng(saved_rng)
        torch.use_deterministic_algorithms(deterministic)
        if device.type == "cpu":
            torch.set_num_threads(old_threads)
