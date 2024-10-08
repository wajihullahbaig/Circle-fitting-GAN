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
    
    return torch.tensor(x).view(-1, 1), torch.tensor(y).view(-1, 1), torch.tensor(theta).view(-1, 1)

def noise_generator(start,end,batch_size):
    z1 = np.random.randint(start, end, size=batch_size).astype(np.float32)/end
    z2 = np.random.randint(start, end, size=batch_size).astype(np.float32)/end
    z3 = 2*np.pi*np.random.randint(start, end, size=batch_size).astype(np.float32)/end
    z = np.column_stack((z1,z2,z3))   
    z = torch.from_numpy(z)    
    return z

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 28),
            nn.ReLU(),
            nn.Linear(28, 3),              
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 48),
            nn.ReLU(),
            nn.Linear(48, 3),  
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    


def train_gan(epochs, dataset_size, batch_size,n_featuresm):
    generator.train()
    discriminator.train()
    
    # create a dataset 
    x_real, y_real,theta_real = generate_real_data(dataset_size)    
    circle_data = torch.cat((x_real, y_real, theta_real), dim=1)
    
    for epoch in range(epochs):    
    # get a batch from dataset
        for batch_count in range(0, dataset_size, batch_size):
            # Create batches using slicing
            real_data = circle_data[batch_count:batch_count + batch_size]
            # Train discriminator
            #z = torch.randn(batch_size, 1) # 1 feature input. Becase we are generating noise using 1 feature.
               
            z = noise_generator(-180,180,batch_size)
            fake_data = generator(z)
            
            real_output = discriminator(real_data)
            fake_output = discriminator(fake_data)
            
            real_loss = criterion(real_output, torch.ones_like(real_output))
            fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
            d_loss = 0.5 * (real_loss + fake_loss)
            
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            fake_samples = generator(z)
            fake_output = discriminator(fake_samples)
            
            
            g_loss = criterion(fake_output, torch.ones_like(fake_output))        
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
   
            if batch_count // batch_size == 1 and epoch %100  == 0 :
                print(f'Epoch: {epoch}, Batch: {batch_count}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

                generator.eval()
                generated = generator(z).detach()
                
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
                generator.train()
                

        
epochs = 50000
batch_size = 128
dataset_size = 360

generator = Generator()
discriminator = Discriminator()

optimizer_g = optim.Adam(generator.parameters(), lr=0.00015)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.00015)
criterion = nn.BCELoss()

n_features = 3
# fixed noise
#z = np.linspace(0 ,np.pi, batch_size).astype(np.float32)
#z = np.tile(z, (n_features, 1)).T
#z = torch.from_numpy(z)


train_gan(epochs,dataset_size, batch_size,n_features)

z = noise_generator(-180,180,batch_size)
generator.eval()
with torch.no_grad():
    generated = generator(z).detach()


plt.figure(figsize=(10,5))
plt.scatter(generated[:,0].numpy(), generated[:,1].numpy(), label='Generated Curve', color='blue')
plt.scatter(*generate_real_data(batch_size), label='Real Data', color='red', alpha=0.5)
plt.title('GAN Generated vs Real Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

