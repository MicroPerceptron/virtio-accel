"""Train the small conditional MNIST VAE used by the inference showcase.

Training needs PyTorch; exported inference needs only NumPy and virtio-accel.
No downloaded Python code or pickle checkpoints are executed.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import urllib.request

import numpy as np

DATA = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def dataset(directory, prefix):
    arrays = []
    for kind in ("images-idx3", "labels-idx1"):
        name = f"{prefix}-{kind}-ubyte.gz"
        path = directory / name
        if not path.exists():
            with urllib.request.urlopen(
                "https://storage.googleapis.com/cvdf-datasets/mnist/" + name, timeout=60
            ) as response:
                data = response.read(16 * 1024 * 1024)
            if hashlib.md5(data).hexdigest() != DATA[name]:
                raise ValueError(f"dataset checksum mismatch: {name}")
            path.write_bytes(data)
        data = path.read_bytes()
        if hashlib.md5(data).hexdigest() != DATA[name]:
            raise ValueError(f"dataset checksum mismatch: {name}")
        raw = gzip.decompress(data)
        magic, count = struct.unpack_from(">II", raw)
        if kind.startswith("images"):
            if (
                magic != 2051
                or struct.unpack_from(">II", raw, 8) != (28, 28)
                or len(raw) != 16 + count * 784
            ):
                raise ValueError("invalid MNIST image dimensions")
            arrays.append(
                np.frombuffer(raw, np.uint8, offset=16)
                .reshape(count, 784)
                .astype(np.float32)
                / 255
            )
        else:
            if magic != 2049 or len(raw) != 8 + count:
                raise ValueError("invalid MNIST labels")
            arrays.append(np.frombuffer(raw, np.uint8, offset=8).astype(np.int64))
    return arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--data", type=Path, default=Path(tempfile.gettempdir()) / "virtio-accel-mnist"
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = p.parse_args()
    if (
        args.output.exists()
        or args.output.suffix != ".npz"
        or args.epochs < 1
        or args.batch_size < 1
    ):
        p.error("output must be a new .npz path and training sizes positive")
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(17)
    torch.set_num_threads(4)
    args.data.mkdir(parents=True, exist_ok=True)
    train_x, train_y = dataset(args.data, "train")
    test_x, test_y = dataset(args.data, "t10k")

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(794, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
            )
            self.mu, self.logvar = nn.Linear(256, 32), nn.Linear(256, 32)
            self.decoder = nn.Sequential(
                nn.Linear(42, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 784),
            )

        def forward(self, images, labels, sample=True):
            labels = F.one_hot(labels, 10).float()
            hidden = self.encoder(torch.cat((images, labels), -1))
            mu, logvar = self.mu(hidden), self.logvar(hidden)
            z = mu + torch.randn_like(mu) * (logvar * 0.5).exp() if sample else mu
            return self.decoder(torch.cat((z, labels), -1)), mu, logvar

    model = VAE().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(train_x).to(args.device)
    y = torch.from_numpy(train_y).to(args.device)
    losses = []
    for epoch in range(args.epochs):
        order = torch.randperm(len(x), device=args.device)
        total = 0.0
        model.train()
        for index in order.split(args.batch_size):
            logits, mu, logvar = model(x[index], y[index])
            recon = F.binary_cross_entropy_with_logits(
                logits, x[index], reduction="sum"
            )
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum()
            loss = (recon + kl) / len(index)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        losses.append(total / len(x))
        print(
            f"epoch {epoch+1}/{args.epochs}: training ELBO loss {losses[-1]:.3f}",
            flush=True,
        )
    model.eval()
    with torch.no_grad():
        tx, ty = torch.from_numpy(test_x).to(args.device), torch.from_numpy(test_y).to(
            args.device
        )
        loss = 0.0
        for start in range(0, len(tx), args.batch_size):
            logits, _, _ = model(
                tx[start : start + args.batch_size],
                ty[start : start + args.batch_size],
                sample=False,
            )
            loss += float(
                F.binary_cross_entropy_with_logits(
                    logits, tx[start : start + args.batch_size], reduction="sum"
                )
            )
    weights = {}
    for i, layer in enumerate((model.decoder[0], model.decoder[2], model.decoder[4])):
        weights[f"w{i}"] = layer.weight.detach().cpu().numpy().T.astype(np.float16)
        weights[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float16)
    metadata = dict(
        schema=1,
        architecture=[42, 512, 512, 784],
        latent_dimensions=32,
        classes=10,
        image_shape=[28, 28],
        model="conditional MNIST VAE decoder",
        seed=17,
        epochs=args.epochs,
        batch_size=args.batch_size,
        training_device=args.device,
        torch_version=torch.__version__,
        training_loss=losses,
        held_out_mean_bce=loss / len(test_x),
        training_examples=60000,
        held_out_examples=10000,
        dataset_url="https://storage.googleapis.com/cvdf-datasets/mnist/",
        dataset_checksums={
            name: dict(
                md5=expected,
                sha256=hashlib.sha256((args.data / name).read_bytes()).hexdigest(),
            )
            for name, expected in DATA.items()
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **weights, metadata=np.array(json.dumps(metadata)))
    print(
        json.dumps(
            dict(
                checkpoint=str(args.output),
                sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                held_out_mean_bce=metadata["held_out_mean_bce"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
