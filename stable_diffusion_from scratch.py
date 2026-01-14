import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import random
from pathlib import Path
import glob

class DigitMaskGenerator:

    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size
        self.base_size = min(image_size[0], image_size[1])
        
        # Initialize font with proper error handling
        try:
            # Try to use a specific system font first
            font_path = self._get_system_font()
            self.font = ImageFont.truetype(font_path, size=self.base_size)
        except Exception as e:
            print(f"Could not load system font: {e}")
            print("Falling back to default font...")
            self.font = ImageFont.load_default()
            
    def _get_system_font(self):

        possible_fonts = [
            # Windows fonts, 
            # Linux fonts,
            # MacOS fonts
        ]
        for font_path in possible_fonts:
            if os.path.exists(font_path):
                return font_path    
        raise FileNotFoundError("No suitable system font found")

    def generate_mask(self, digit):
        large_size = (self.base_size * 2, self.base_size * 2)
        large_image = Image.new('L', large_size, 0) 
        large_draw = ImageDraw.Draw(large_image)
        
        # convert digit to string and get its size for centering
        digit_str = str(digit)
        bbox = large_draw.textbbox((0, 0), digit_str, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # center position
        x = (large_size[0] - text_width) // 2
        y = (large_size[1] - text_height) // 2
        
        large_draw.text((x, y), digit_str, fill=255, font=self.font)

        image = large_image.resize(self.image_size, Image.Resampling.LANCZOS)
        tensor = transforms.ToTensor()(image)
        
        # slight blur to help blend with landscape
        tensor = transforms.GaussianBlur(kernel_size=3, sigma=0.5)(tensor)
        
        return tensor

class LandscapeDataset(Dataset):

    def __init__(self, landscape_paths, transform=None):
        self.landscape_paths = landscape_paths
        self.transform = transform
        self.mask_generator = DigitMaskGenerator()
        
    def __len__(self):
        return len(self.landscape_paths)
    
    def __getitem__(self, idx):
        landscape = Image.open(self.landscape_paths[idx]).convert('RGB')
        if self.transform:
            landscape = self.transform(landscape)
            
        # generate random digit for mask
        digit = random.randint(0, 9)
        mask = self.mask_generator.generate_mask(digit)
        
        return landscape, mask, digit

class DigitConditionedUNet(nn.Module):
    # U-Net architecture
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(DigitConditionedUNet, self).__init__()
        self.bottleneck_features = features * 16
        # downsampling
        self.enc1 = self.conv_block(in_channels + 1, features)
        self.enc2 = self.conv_block(features, features * 2)
        self.enc3 = self.conv_block(features * 2, features * 4)
        self.enc4 = self.conv_block(features * 4, features * 8)
        
        # bottleneck
        self.bottleneck = self.conv_block(features * 8, self.bottleneck_features)
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.bottleneck_features),
            nn.SiLU(),
            nn.Linear(self.bottleneck_features, self.bottleneck_features)
        )
        
        # upsampling
        self.up4 = nn.ConvTranspose2d(self.bottleneck_features, features * 8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(features * 16, features * 8)  
        
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(features * 8, features * 4)   
        
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(features * 4, features * 2)   
        
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(features * 2, features)    
        
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, digit_mask, t):

        # combine input image and digit mask
        x = torch.cat([x, digit_mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        bottleneck = self.bottleneck(F.max_pool2d(e4, 2))

        t_emb = self.time_embedding(t.unsqueeze(1)) 
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)

        bottleneck = bottleneck + t_emb.expand_as(bottleneck)

        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
    
class DiffusionModel:

    def __init__(self, num_timesteps=1000, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alpha = (1 - self.beta).to(device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)

        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha)
        self.sqrt_recip_alpha = torch.sqrt(1 / self.alpha)
        self.posterior_variance = self.beta * (1 - self.alpha_bar.roll(1)) / (1 - self.alpha_bar)
        self.posterior_variance[0] = self.beta[0]  
        
    def forward_diffusion(self, x, t):

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(x, device=x.device)

        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise
    
    def reverse_diffusion(self, model, x, digit_mask, t):

        timestep = t.item()
        predicted_noise = model(x, digit_mask, t.float() / self.num_timesteps)
        alpha = self.alpha[timestep]
        alpha_bar = self.alpha_bar[timestep]
        beta = self.beta[timestep]
        x_0_pred = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise)
        
        if timestep > 0:
            noise = torch.randn_like(x, device=x.device)
            variance = torch.sqrt(self.posterior_variance[timestep])
            x_prev = x_0_pred + variance * noise
        else:
            x_prev = x_0_pred
        
        return x_prev
    
    def reverse_diffusion_loop(self, model, x, digit_mask):
        model.eval()
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                timestep = torch.tensor([t], device=x.device)
                x = self.reverse_diffusion(model, x, digit_mask, timestep)
                if t % 100 == 0:
                    print(f"Denoising step {self.num_timesteps - t}/{self.num_timesteps}")
        
        return x


def train_model(model, diffusion, dataloader, num_epochs, device):

    progress_dir = os.path.join('Gan_project', 'training_progress')
    os.makedirs(progress_dir, exist_ok=True)

    print(f"Training on device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    total_batches = len(dataloader)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (landscape, mask, digit) in enumerate(dataloader):
            try:
                landscape = landscape.to(device)
                mask = mask.to(device)
                t = torch.randint(0, diffusion.num_timesteps, (landscape.shape[0],), 
                                device=device)

                noisy_landscape, noise = diffusion.forward_diffusion(landscape, t)
                timestep_values = t.float() / diffusion.num_timesteps
                noise_pred = model(noisy_landscape, mask, timestep_values)
                loss = criterion(noise_pred, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], "
                          f"Batch [{batch_idx}/{total_batches}], "
                          f"Loss: {loss.item():.4f}")
                    
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}:")
                print(f"Landscape device: {landscape.device}")
                print(f"Mask device: {mask.device}")
                print(f"Model device: {next(model.parameters()).device}")
                print(f"Error message: {str(e)}")
                raise e

        avg_epoch_loss = epoch_loss / total_batches
        print(f"\nEpoch [{epoch+1}/{num_epochs}] completed, "
              f"Average Loss: {avg_epoch_loss:.4f}")

        print(f"\nGenerating sample images for epoch {epoch + 1}...")

        model.eval()
        with torch.no_grad():
            for digit in range(10):
                try:
                    generated = generate_landscape_with_digit(
                        model, diffusion, digit, device, img_size=(256, 256)
                    )

                    img = transforms.ToPILImage()(generated.squeeze(0).cpu())
                    filename = f'epoch_{epoch+1:03d}digit{digit}.png'
                    save_path = os.path.join(progress_dir, filename)
                    img.save(save_path)
                    print(f"Saved sample with digit {digit} for epoch {epoch + 1}")
                
                except Exception as e:
                    print(f"Error generating image for digit {digit} at epoch {epoch + 1}: {str(e)}")
                    continue
        model.train()
        # save model checkpoint every epoch
        checkpoint_path = os.path.join('Gan_project', f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        print(f"Saved model checkpoint for epoch {epoch + 1}\n")

def generate_landscape_with_digit(model, diffusion, digit, device, img_size=(256, 256)):

    model.eval()
    
    with torch.no_grad():
        mask_generator = DigitMaskGenerator(img_size)
        digit_mask = mask_generator.generate_mask(digit)
        digit_mask = digit_mask.unsqueeze(0).to(device)  
        x = torch.randn((1, 3, *img_size), device=device)

        for t in reversed(range(diffusion.num_timesteps)):
            timesteps = torch.tensor([t], device=device)
            x = diffusion.reverse_diffusion(
                model,
                x,
                digit_mask,
                timesteps
            )
        x = (x.clamp(-1, 1) + 1) / 2
        
    return x

def save_generated_image(generated_tensor, output_path):

    pil_image = transforms.ToPILImage()(generated_tensor.squeeze(0).cpu())
    pil_image.save(output_path)

def setup_training(landscape_dir, batch_size=16, image_size=(256, 256)):
    if not os.path.exists(landscape_dir):
        raise FileNotFoundError(f"Directory not found: {landscape_dir}")
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    landscape_paths = []
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        pattern = os.path.join(landscape_dir, f'*{ext}')
        landscape_paths.extend(glob.glob(pattern))
    if not landscape_paths:
        raise FileNotFoundError(
            f"No JPEG images found in {landscape_dir}. "
            "Please ensure your images are in JPEG format and the path is correct."
        )
    
    print(f"Found {len(landscape_paths)} images for training")
    
    try:
        dataset = LandscapeDataset(landscape_paths, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(), 
            drop_last=True  
        )

        test_batch = next(iter(dataloader))
        print(f"Successfully verified data loading. Batch shape: {test_batch[0].shape}")
        
        return dataloader
        
    except Exception as e:
        raise RuntimeError(f"Error setting up data loading: {str(e)}")
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    image_size = (256, 256)  
    batch_size = 16          
    num_epochs = 100         

    landscape_dir = "your_path/landscapes/*.jpg"
    output_dir = "your_path/generated_landscapes"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for landscape images in: {landscape_dir}")

    model = DigitConditionedUNet().to(device)
    diffusion = DiffusionModel(num_timesteps=1000)

    try:
        base_landscape_dir = os.path.dirname(landscape_dir)
        dataloader = setup_training(base_landscape_dir, batch_size, image_size)
        print(f"Successfully loaded {len(dataloader.dataset)} landscape images")
    except FileNotFoundError as e:
        print(f"Error: Could not find landscape images at {landscape_dir}")
        return
    except Exception as e:
        print(f"Unexpected error setting up training: {e}")
        return
    try:
        train_model(model, diffusion, dataloader, num_epochs, device)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    model_path = os.path.join('your_path', 'landscape_digit_model.pth')
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    print("Generating example landscapes with hidden digits...")
    for digit in range(10):
        try:
            generated = generate_landscape_with_digit(model, diffusion, digit, device, image_size)
            output_path = os.path.join(output_dir, f'landscape_with_{digit}.png')
            save_generated_image(generated, output_path)
            print(f"Generated landscape with hidden digit {digit} saved to {output_path}")
        except Exception as e:
            print(f"Error generating landscape with digit {digit}: {e}")
            continue

if __name__ == '__main__':
    main()