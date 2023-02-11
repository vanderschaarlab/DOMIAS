# stdlib
import argparse
import os
from typing import Any, Generator, Optional, Tuple

# third party
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as tf
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_idx", default=None, type=int)
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--rep_dim", type=int, default=128)
parser.add_argument("--ae_epoch", type=int, default=10)  # 1000
parser.add_argument("--dcgan_epoch", type=int, default=10)  # 1000
parser.add_argument("--training_size", type=int, default=1000)


args = parser.parse_args()

GPU_IDX = args.gpu_idx
SEED = args.seed
LATENT_REPRESENTATION_DIM = args.rep_dim
AE_EPOCH = args.ae_epoch

alias = f"{SEED}_{GPU_IDX}_{LATENT_REPRESENTATION_DIM}_tsz{args.training_size}"


os.system(f"mkdir -p debug_train_{args.training_size}/debug_train_{args.training_size}")
os.system("mkdir -p celebA_representation")

for i in range(int(args.training_size / 1000)):
    os.system(
        f"cp -r data/CelebA-Data/subset_5000/subset_5000/00{i}* debug_train_{args.training_size}/debug_train_{args.training_size}/"
    )


os.system("mkdir -p debug_ref/debug_ref")
for i in range(5):
    os.system(
        f"cp -r data/CelebA-Data/subset_5k15k/subset_5k15k/00{5+i}* debug_ref/debug_ref/"
    )
for i in range(5):
    os.system(
        f"cp -r data/CelebA-Data/subset_5k15k/subset_5k15k/01{i}* debug_ref/debug_ref/"
    )


os.system(f"mkdir -p debug_test_{args.training_size}/debug_test_{args.training_size}")
for i in range(int(args.training_size / 1000)):
    os.system(
        f"cp -r data/CelebA-Data/subset_15k20k/subset_15k20k/01{5+i}* debug_test_{args.training_size}/debug_test_{args.training_size}/"
    )


dirc = f"debug_train_{args.training_size}"
device = torch.device(f"cuda:{GPU_IDX}" if torch.cuda.is_available() else "cpu")

print(device)


workers = 8
batch_size = 128
image_size = 64
ms = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


dataset = ImageFolder(
    dirc,
    transform=tf.Compose(
        [
            tf.Resize(image_size),
            tf.CenterCrop(image_size),
            tf.ToTensor(),
            tf.Normalize(*ms),
        ]
    ),
)
# Create the dataloader
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)


def denormalized(img: np.ndarray) -> np.ndarray:
    return img * ms[1][0] + ms[0][0]


def deviceLoaderfunc(data: Any, device: Any) -> torch.Tensor:
    if isinstance(data, (list, tuple)):
        return [deviceLoaderfunc(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl: Any, device: Any) -> None:
        self.dl = dl
        self.device = device

    def __iter__(self) -> Generator:
        for b in self.dl:
            yield deviceLoaderfunc(b, self.device)

    def __len__(self) -> int:
        return len(self.dl)


dataloader = DeviceDataLoader(dataloader, device)


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class View(nn.Module):
    def __init__(self, size: Tuple) -> None:
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim: int = 10, nc: int = 3) -> None:
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, z_dim * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self) -> None:
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distributions = self._encode(x)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim: int = 10, nc: int = 1) -> None:
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),  # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 256 * 4 * 4)),  # B, 512
            nn.Linear(256 * 4 * 4, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim * 2),  # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256 * 4 * 4),  # B, 512
            nn.ReLU(True),
            View((-1, 256, 4, 4)),  # B,  32,  4,  4
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, nc, 4, 2, 1),  # B,  nc, 64, 64
            nn.Tanh(),
        )
        self.weight_init()

    def weight_init(self) -> None:
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distributions = self._encode(x)
        mu = distributions[:, : self.z_dim]
        logvar = distributions[:, self.z_dim :]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def kaiming_init(m: Any) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m: Any, mean: torch.Tensor, std: torch.Tensor) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def reconstruction_loss(
    x: torch.Tensor, x_recon: torch.Tensor, distribution: torch.Tensor
) -> Optional[torch.Tensor]:
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == "bernoulli":
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, size_average=False
        ).div(batch_size)
    elif distribution == "gaussian":
        # x_recon = F.tanh(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(
    mu: torch.Tensor, logvar: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


beta_vae = BetaVAE_B(z_dim=32, nc=3).to(device)


# def fit(epochs, lr, index=1):
lr = 0.002
epochs = args.dcgan_epoch
OBJ = "H"
C_max = 25
C_stop_iter = 1e5

torch.cuda.empty_cache()
beta_vae_optimizer = torch.optim.Adam(beta_vae.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(epochs):
    for r_images, _ in tqdm(dataloader):
        x_recon, mu, logvar = beta_vae(r_images)
        recon_loss = reconstruction_loss(r_images, x_recon, "gaussian")

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        beta_vae_loss = recon_loss + 1.0 * total_kld
    if recon_loss is None:
        raise RuntimeError("invalid recon_loss")

    if beta_vae_loss is None:
        raise RuntimeError("invalid recon_loss")

    beta_vae_optimizer.zero_grad()
    beta_vae_loss.backward()
    beta_vae_optimizer.step()

    print(
        "epoch:",
        epoch + 1,
        "Loss:",
        beta_vae_loss.item(),
        "recon loss:",
        recon_loss.item(),
    )
plt.imshow(x_recon[0].cpu().detach().permute(1, 2, 0))
plt.show()

"""Auto-Encoder Training"""

encoder = nn.Sequential(
    # 3 x 64 x 64
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # 64 x 32 x 32
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # 128 x 16 x 16
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # 256 x 8 x 8
    nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),
    # 512 x 4 x 4
    nn.Conv2d(
        256, LATENT_REPRESENTATION_DIM, kernel_size=4, stride=1, padding=0, bias=False
    ),
    # 1 x 1 x 1
    # nn.Flatten(),
    nn.Tanh(),
).to(device)

decoder = nn.Sequential(
    nn.ConvTranspose2d(
        LATENT_REPRESENTATION_DIM, 256, kernel_size=4, stride=1, padding=0, bias=False
    ),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    # 512 x 4 x 4 upsampled data
    nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    # 256 x 8 x 8
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    # 128 x 16 x 16
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    # 64 x 32 x 32
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # 3 x 64 x 64
).to(device)


def fit_AE(epochs: int, lr: float, index: int = 1) -> None:

    torch.cuda.empty_cache()
    enc_dec_optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": lr, "betas": (0.5, 0.999)},
            {"params": decoder.parameters(), "lr": lr, "betas": (0.5, 0.999)},
        ]
    )
    # dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for r_images, _ in tqdm(dataloader):
            #             loss_d, r_score, g_score = train_discriminator(r_images, d_optimizer)
            #             loss_g = train_generator(g_optimizer)

            r_recover = decoder(encoder(r_images))
            loss = F.mse_loss(r_recover, r_images)
            loss.backward()
            enc_dec_optimizer.step()

        print("epoch:", epoch, " loss:", loss.item())


fit_AE(2, 0.0002)

HIDDEN = LATENT_REPRESENTATION_DIM
N_SYNTH = 10000
synth_representation = np.zeros((N_SYNTH, HIDDEN))
if N_SYNTH > 50:
    for i in range(int(N_SYNTH / 50)):
        latent_noise = torch.randn(50, beta_vae.z_dim, device=device)
        g_images = beta_vae.decoder(latent_noise)
        synth_representation[i * 50 : (i + 1) * 50] = (
            encoder(g_images).cpu().detach().squeeze().numpy()
        )


np.save(
    f"celebA_representation/AISTATS_betavae_repres_synth_{alias}", synth_representation
)

N_REAL = len(dataset)
real_representation = np.zeros((N_REAL, HIDDEN))
i = 0
for r_images, _ in tqdm(dataloader):
    local_bs = min(batch_size, len(r_images))
    real_representation[i * batch_size : i * batch_size + local_bs] = (
        encoder(r_images).cpu().detach().squeeze().numpy()
    )
    i += 1

np.save(
    f"celebA_representation/AISTATS_betavae_repres_real_{alias}", real_representation
)


ref_dataset = ImageFolder(
    "debug_ref",
    transform=tf.Compose(
        [
            tf.Resize(image_size),
            tf.CenterCrop(image_size),
            tf.ToTensor(),
            tf.Normalize(*ms),
        ]
    ),
)
# Create the dataloader
ref_dataloader = DataLoader(
    ref_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)
ref_dataloader = DeviceDataLoader(ref_dataloader, device)


N_REF = len(ref_dataset)

ref_representation = np.zeros((N_REF, HIDDEN))
i = 0
for r_images, _ in tqdm(ref_dataloader):
    local_bs = min(batch_size, len(r_images))
    ref_representation[i * batch_size : i * batch_size + local_bs] = (
        encoder(r_images).cpu().detach().squeeze().numpy()
    )
    i += 1


np.save(f"celebA_representation/AISTATS_betavae_repres_ref_{alias}", ref_representation)

test_dataset = ImageFolder(
    f"debug_test_{args.training_size}",
    transform=tf.Compose(
        [
            tf.Resize(image_size),
            tf.CenterCrop(image_size),
            tf.ToTensor(),
            tf.Normalize(*ms),
        ]
    ),
)
# Create the dataloader
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)
test_dataloader = DeviceDataLoader(test_dataloader, device)

N_TEST = len(test_dataset)

test_representation = np.zeros((N_TEST, HIDDEN))
i = 0
for r_images, _ in tqdm(test_dataloader):
    local_bs = min(batch_size, len(r_images))
    test_representation[i * batch_size : i * batch_size + local_bs] = (
        encoder(r_images).cpu().detach().squeeze().numpy()
    )
    i += 1

np.save(
    f"celebA_representation/AISTATS_betavae_repres_test_{alias}", test_representation
)
