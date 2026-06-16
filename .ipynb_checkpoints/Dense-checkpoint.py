import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# DM with only MLP https://github.com/schinger/DiffusionModel  (IMPLEMENTATION INCOMPLETE)
class Diffusor(nn.Module):
    def __init__(
        self,
        base_channel_size:int,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        dtype=torch.float32
    ):
        super().__init()
        self.num_timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=dtype)
        self.alphas = 1.0 - self.betas

        # multiplicative noising schedule
        self.alphas_cp = torch.cumprod(self.alphas, dim=0)
        self.alphas_cp_prev = torch.zeros_like(self.alphas_cp)
        self.alphas_cp_prev[1:] = self.alphas_cp[:-1] #offset 1
        self.sqrt_alphas_cp = self.alphas_cp**0.5
        self.sqrt_1malphas_cp = (1 - self.alphas_cp)**0.5

        # noised step metrics
        self.sqrt_inv_alphas = torch.sqrt(1 / self.alphas)
        self.noise_coeff = self.betas / self.sqrt_1malphas_cp
        self.variance = self.betas * (1 - self.alphas_cp_prev)/( 1 - self.alphas_cp)

    def add_noise(self, x_start, x_noise, timesteps):
        # compute x_t for DDPM
        s1 = self.sqrt_alphas_cp[timesteps][:,None]
        s2 = self.sqrt_1malphas_cp[timesteps][:,None]
        return s1 * x_start + s2 * x_noise # time-weighted mix from x to noise

    def sample_step(self, model_output, timestep, sample):
        # compute x_{t - 1} for DDPM
        s1 = self.sqrt_inv_alphas[timestep][:,None]
        s2 = self.noise_coeff[timestep][:,None]
        s3 = self.variance[timestep][:,None]**0.5
        noise = torch.randn_like(model_output)
        return s1*(sample - s2*model_output) + s3*noise

    def __len__(self):
        return self.num_timesteps

    def train(self, dataloader, model, optimizer=None, scheduler=None, num_epochs=200):
        if (optimizer is None): optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        if (scheduler is None): scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step/(num_epochs * len(x)))
        for epoch in range(num_epochs):
            for x in dataloader:
                # add random quantity of noise to data
                t = torch.randint(0, self.num_timesteps, (batch.shape[0],)).long() #(B,)
                noise = torch.randn_like(batch) #B,N
                x_t = self.add_noise(batch, noise, t)
                # predict noise added to batch of samples
                noise_p = model(x, x_t, t)
                
                loss = F.mse_loss(noise_p, noise)
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        return model
    def train_batch(self, x, model, optimizer=None, scheduler=None, num_epochs=200):
        return self.train([x], model, optimizer, scheduler, num_epochs)

    def reverse(self, x, model):
        model.eval()
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            x = self.sample_step(x, model(x, t_batch), t, x)
        return x

# --- MLP components

# embed discrete diffusion timestep to smooth higher-dimensional function for stability
class SinusoidalEmbedding(nn.Module):  
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        self.emb = nn.Parameter(emb, requires_grad=False)
    def forward(self, x: torch.Tensor):
        emb = x * self.scale * self.emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class MLPBlock(nn.Module):
    def __init__(self, size:int):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(size),     # normalization
            nn.Linear(size, size),  # feedforward
            nn.GELU()               # activation
        )
    def forward(self, x):
        return x + self.block(x)
        
class MLP(nn.Module):
    def __init__(self, hidden_size:int = 128, hidden_layers:int = 3, data_size:int=64, emb_size:int = 128):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(emb_size)
        self.concat_size = data_size + emb_size
        self.data_size = data_size

        layers = [nn.Linear(self.concat_size, hidden_size, nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(MLPBlock(hidden_size))
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.Linear(hidden_size, self.data_size))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_emb(t.reshape(-1,1))
        x = torch.cat((x, t_emb), dim = -1)
        x = self.mlp(x)
        return x


        

# -- autoencoder

class Autoencoder(nn.Module):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

# affine layer ~ Ax + B (from ResMLP @ https://github.com/rishikksh20/ResMLP-pytorch/blob/master/resmlp.py) 
#class Aff(nn.Module):
#    def __init__(self, dim):
#        super().__init__()
#        self.ab = nn.Parameter(torch.ones(1,1,dim,2))
#    def forward(self, x):
#        return x * self.ab[..., 0] + self.ab[..., 1]



# --- CNN components

class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image.
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network

        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """Decoder.

        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network

        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x