# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:13:28 2024

@author: Acer
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)
np.random.seed(0)

# Function to generate real data points based on the function y = mx^2 + c
def generate_real_data(n):
    theta = np.linspace(0, 2*np.pi, n).astype(np.float32)
    radius = 1
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)    
    
    return torch.tensor(x).view(-1, 1), torch.tensor(y).view(-1, 1)

def noise_generator(start,end,batch_size):
    z1 = np.random.randint(start, end, size=batch_size).astype(np.float32)/end
    z2 = np.random.randint(start, end, size=batch_size).astype(np.float32)/end
    z = np.column_stack((z1,z2))   
    z = torch.from_numpy(z)    
    return z

# Define the VAE Encoder
class VAEEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))  # Apply ReLU activation
        z_mean = self.fc21(h1)         # Mean of the latent space
        z_log_var = self.fc22(h1)      # Log variance of the latent space
        return z_mean, z_log_var

# Define the VAE Decoder (Generator)
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h3 = torch.relu(self.fc3(z))   # Apply ReLU activation
        return torch.tanh(self.fc4(h3))  # Output layer with tanh activation

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, n_features,laten_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(n_features, laten_dim)
        self.fc2 = nn.Linear(laten_dim, 1)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))   # Apply ReLU activation
        return torch.sigmoid(self.fc2(h1))  # Output layer with sigmoid activation
    
# Reparameterization trick for VAE
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std    


def train_gan(vae_encoder,vae_decoder,discriminator,optimizer_vae,optimizer_disc, epochs, dataset_size, batch_size,n_featuresm):
    vae_encoder.train()
    vae_decoder.train()
    discriminator.train()
    
    # create a dataset 
    x_real, y_real = generate_real_data(dataset_size)    
    circle_data = torch.cat((x_real, y_real), dim=1)
    batch_count = 0
    for epoch in range(epochs):    
    # get a batch from dataset
        for batch_count in range(0, dataset_size, batch_size):            
            # Create batches using slicing
            real_data = circle_data[batch_count:batch_count + batch_size]
            # real_data is already in the correct shape (batch_size, input_dim)
            # Train Discriminator
            optimizer_disc.zero_grad()
            mu, logvar = vae_encoder(real_data)
            assert not torch.isnan(mu).any()
            assert not torch.isnan(logvar).any()

            z = reparameterize(mu, logvar)
            fake_data = vae_decoder(z)

            disc_real = discriminator(real_data)
            disc_fake = discriminator(fake_data.detach())
            loss_disc = -torch.mean(torch.log(disc_real) + torch.log(1 - disc_fake))
            loss_disc.backward()
            # Gradient clipping for discriminator to avoid nans
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_disc.step()

            # Train VAE-GAN
            optimizer_vae.zero_grad()
            disc_fake = discriminator(fake_data)
            loss_vae_gan = -torch.mean(torch.log(disc_fake))  # Adversarial loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
            loss_vae_total = loss_vae_gan + kl_loss
            loss_vae_total.backward()
            # Gradient clipping for encoder and decoder to avoid nans
            torch.nn.utils.clip_grad_norm_(vae_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(vae_decoder.parameters(), max_norm=1.0)
            optimizer_vae.step()
            
            if batch_count // batch_size == 1 and epoch %100  == 0 :
                print(f'Epoch: {epoch}, VAE Loss: {loss_vae_gan.item()}, Discriminator Loss: {loss_disc.item()}, KL Loss {kl_loss.item()}')

                vae_decoder.eval()
                generated = vae_decoder(z).detach()
                
                plt.figure(figsize=(10,5))
                plt.scatter(*generate_real_data(dataset_size), label='Real Data', color='red', alpha=0.5)            
                plt.scatter(generated[:,0].numpy(), generated[:,1].numpy(), label='Generated Curve', color='blue')
                plt.title('GAN Generated vs Real Data')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.legend()
                #plt.show()
                plt.savefig(f'images/{epoch}_{batch_count}.jpg', bbox_inches='tight')
                plt.clf()
                plt.close()
                vae_decoder.train()


                


n_features = 2
hidden_dim = 100
latent_dim = 50

vae_encoder = VAEEncoder(n_features, hidden_dim, latent_dim)
vae_decoder = VAEDecoder(latent_dim, hidden_dim, output_dim=n_features)
discriminator = Discriminator(n_features,latent_dim)
        
epochs = 50000
batch_size = 128
dataset_size = 360


optimizer_vae = optim.Adam(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=0.0001)
optimizer_disc = optim.Adam(discriminator.parameters(), lr=0.0001)


train_gan(vae_encoder,vae_decoder,discriminator,optimizer_vae,optimizer_disc,epochs,dataset_size, batch_size,n_features)

# Lets take sample from training data to generate synthetic data
vae_encoder.eval()
vae_decoder.eval()
with torch.no_grad():
    z = noise_generator(-180,180,dataset_size)    
    mu, logvar = vae_encoder(z)
    z = reparameterize(mu, logvar)
    generated = vae_decoder(z).detach()



plt.figure(figsize=(10,5))
plt.scatter(generated[:,0].numpy(), generated[:,1].numpy(), label='Generated Curve', color='blue')
plt.scatter(*generate_real_data(batch_size), label='Real Data', color='red', alpha=0.5)
plt.title('GAN Generated vs Real Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()




