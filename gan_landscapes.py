import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision
from torchvision.utils import save_image
from PIL import Image
import glob
import os
import time

# custom landscape dataset
class ImageDataset(Dataset):
    def __init__(self, path, image_size=256):  # Updated default size
        self.files = sorted(glob.glob(path))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.files)


class Generator(nn.Module):
    def __init__(self, noise_dim, channels=64):
        super(Generator, self).__init__()
        
        self.init_size = 16  
        self.noise_dim = noise_dim
        self.channels = channels

        self.l1 = nn.Linear(noise_dim, 8*channels*self.init_size*self.init_size)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(8*channels),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8*channels, 8*channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(8*channels, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8*channels, 4*channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(4*channels, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4*channels, 2*channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(2*channels, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*channels, channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(channels, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 8*self.channels, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, kernel_size=4):
            block = [
                nn.Conv2d(in_filters, out_filters, kernel_size, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, channels, bn=False),
            *discriminator_block(channels, channels*2),
            *discriminator_block(channels*2, channels*4),
            *discriminator_block(channels*4, channels*8),
            *discriminator_block(channels*8, channels*16),
        )

        # calculate the size of output from convo layers
        ds_size = 8
        self.adv_layer = nn.Sequential(
            nn.Linear(channels*16 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        features = self.model(img)
        features = features.view(features.size(0), -1)
        validity = self.adv_layer(features)
        return validity


def save_image(epoch, noise, generator, save_dir):
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(noise)
    
    os.makedirs(save_dir, exist_ok=True)
    # save 3 generated images separately
    for i in range(3):
        torchvision.utils.save_image(gen_imgs[i], f"{save_dir}/sample_epoch_{epoch}_img_{i}.png", normalize=True)
    
    generator.train()


def train(epochs, batch_size, learning_rate, save_freq, input_path, save_dir, image_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = ImageDataset(input_path, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset loaded. Total batches: {len(dataloader)}")

    generator = Generator(noise_dim=100).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss()
    fixed_noise = torch.randn(3, 100, device=device)

    print("Starting training...")
    start_time = time.time()
    
    try:
        for epoch in range(epochs):
            for i, imgs in enumerate(dataloader):
                real_imgs = imgs.to(device)
                batch_size = real_imgs.size(0)

                valid = torch.ones(batch_size, 1, device=device)
                fake = torch.zeros(batch_size, 1, device=device)

                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, 100, device=device)
                gen_imgs = generator(z)
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                if i % 50 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                          f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

                if i % 5 == 0:
                    torch.cuda.empty_cache()

            if epoch % save_freq == 0:
                save_image(epoch, fixed_noise, generator, save_dir)
                print(f"Images saved for epoch {epoch} in {save_dir}")

            print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")
            print(f"Time elapsed: {(time.time() - start_time) / 60:.2f} minutes")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    finally:
        torch.save(generator.state_dict(), f"{save_dir}/generator.pth")
        print("Training completed. Generator saved.")

if __name__ == "__main__":
    train(epochs=1001, batch_size=32, learning_rate=0.0002, save_freq=50, 
          input_path="Gan_project/landscapes/*.jpg", 
          save_dir="Gan_project/generated_images_256",
          image_size=256)