import argparse
from pathlib import Path

import torch
import yaml

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.seed import set_seed
from utils.logging import CSVLogger
from utils.plots import plot_series, savefig
from data.synthetic import make_overlapping_gmm
from data.mnist import get_mnist_ssl
from data.cifar10 import get_cifar10_ssl
from methods.em_gmm import run_em
from methods.self_training import run_self_training
from methods.fixmatch import run_fixmatch
from methods.mean_teacher import run_mean_teacher
from methods.hybrid_teacher_threshold import run_hybrid
from models.small_cnn import SmallCNN
from models.resnet18 import build_resnet18


def build_model(name: str):
    if name == "small_cnn":
        return SmallCNN()
    if name == "resnet18":
        return build_resnet18()
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 0))

    out_dir = Path(cfg.get("output_dir", "outputs"))
    logs_dir = out_dir / "logs"
    figs_dir = out_dir / "figures"

    exp = cfg.get("experiment")

    if exp == "em_gmm_overlap":
        rng = torch.Generator().manual_seed(cfg["seed"])
        import numpy as np
        np_rng = np.random.default_rng(cfg["seed"])
        data = make_overlapping_gmm(
            np_rng,
            n_samples=cfg["data"]["n_samples"],
            means=cfg["data"]["means"],
            cov_scale=cfg["data"]["cov_scale"],
            overlap=cfg["data"]["overlap"],
        )
        history = run_em(
            data.X,
            init_means=cfg["em"]["init_means"],
            iters=cfg["em"]["iters"],
            sigma=cfg["em"]["sigma"],
        )
        logger = CSVLogger(logs_dir, "em_gmm")
        for i, (means, ll, ent) in enumerate(zip(history.means, history.log_likelihood, history.entropy)):
            logger.log({"iter": float(i), "log_likelihood": ll, "entropy": ent})
        logger.flush()

        plot_series(range(len(history.log_likelihood)), {"log_likelihood": history.log_likelihood}, "EM objective", "ll")
        savefig(figs_dir / "em_loglik.png")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg["data"]["dataset"] == "mnist":
        loaders = get_mnist_ssl(
            cfg["data"]["data_dir"],
            cfg["data"]["labeled_per_class"],
            cfg["data"]["batch_size"],
            cfg["data"]["num_workers"],
            cfg["seed"],
        )
    else:
        loaders = get_cifar10_ssl(
            cfg["data"]["data_dir"],
            cfg["data"]["labeled_per_class"],
            cfg["data"]["batch_size"],
            cfg["data"]["num_workers"],
            cfg["seed"],
        )

    model = build_model(cfg["model"]["name"])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["train"]["lr"], momentum=0.9, weight_decay=cfg["train"]["weight_decay"])

    if exp == "selftrain_mnist":
        result = run_self_training(
            model,
            loaders.labeled,
            loaders.unlabeled,
            loaders.unlabeled_eval,
            loaders.test,
            optimizer,
            device,
            rounds=cfg["train"]["rounds"],
            threshold=cfg["train"]["threshold"],
            use_soft=cfg["train"]["use_soft"],
            max_unlabeled_per_round=cfg["train"]["max_unlabeled_per_round"],
        )
        logger = CSVLogger(logs_dir, "selftrain")
        for row in result.history:
            logger.log(row)
        logger.flush()
        plot_series(range(len(result.history)), {"test_acc": [r["test_acc"] for r in result.history]}, "Self-training", "test_acc")
        savefig(figs_dir / "selftrain_test_acc.png")
        return

    if exp == "fixmatch_cifar10":
        result = run_fixmatch(
            model,
            loaders.labeled,
            loaders.unlabeled,
            loaders.unlabeled_eval,
            loaders.test,
            optimizer,
            device,
            epochs=cfg["train"]["epochs"],
            tau=cfg["train"]["tau"],
            lambda_u=cfg["train"]["lambda_u"],
        )
        logger = CSVLogger(logs_dir, "fixmatch")
        for row in result.history:
            logger.log(row)
        logger.flush()
        plot_series(range(len(result.history)), {"test_acc": [r["test_acc"] for r in result.history]}, "FixMatch", "test_acc")
        savefig(figs_dir / "fixmatch_test_acc.png")
        return

    if exp == "mean_teacher_cifar10":
        teacher = build_model(cfg["model"]["name"])
        result = run_mean_teacher(
            model,
            teacher,
            loaders.labeled,
            loaders.unlabeled,
            loaders.test,
            optimizer,
            device,
            epochs=cfg["train"]["epochs"],
            ema_decay=cfg["train"]["ema_decay"],
            lambda_u=cfg["train"]["lambda_u"],
        )
        logger = CSVLogger(logs_dir, "mean_teacher")
        for row in result.history:
            logger.log(row)
        logger.flush()
        plot_series(range(len(result.history)), {"test_acc": [r["test_acc"] for r in result.history]}, "Mean Teacher", "test_acc")
        savefig(figs_dir / "mean_teacher_test_acc.png")
        return

    if exp == "hybrid_teacher_threshold":
        teacher = build_model(cfg["model"]["name"])
        result = run_hybrid(
            model,
            teacher,
            loaders.labeled,
            loaders.unlabeled,
            loaders.unlabeled_eval,
            loaders.test,
            optimizer,
            device,
            epochs=cfg["train"]["epochs"],
            ema_decay=cfg["train"]["ema_decay"],
            tau=cfg["train"]["tau"],
            lambda_u=cfg["train"]["lambda_u"],
        )
        logger = CSVLogger(logs_dir, "hybrid")
        for row in result.history:
            logger.log(row)
        logger.flush()
        plot_series(range(len(result.history)), {"test_acc": [r["test_acc"] for r in result.history]}, "Hybrid", "test_acc")
        savefig(figs_dir / "hybrid_test_acc.png")
        return

    raise ValueError(f"Unknown experiment: {exp}")


if __name__ == "__main__":
    main()
