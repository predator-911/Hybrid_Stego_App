import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import re
from diffusers import StableDiffusionPipeline

# Import AES encryption and encoding tables from utils
from stego_utils import (
    AESCipher,
    OBJECT_TABLE,
    COLOR_TABLE,
    STYLE_TABLE,
    HIGHLIGHT_TABLE,
    INV_OBJECT_TABLE,
    INV_COLOR_TABLE,
    INV_STYLE_TABLE,
    INV_HIGHLIGHT_TABLE,
    encode_binary_to_prompt,
    decode_prompt_to_binary
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Part 1: GAN-based Steganography Models
class RGBStegoEncoder(nn.Module):
    def __init__(self):
        super(RGBStegoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.message_processor = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
        )

        self.r_channel_processor = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.g_channel_processor = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.b_channel_processor = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        self.spatial_transformer = nn.Sequential(
            nn.Conv2d(3 + 8, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, image, message):
        batch_size = image.size(0)
        x = F.leaky_relu(self.bn1(self.conv1(image)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x_res1 = x
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x_res2 = x
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2) + F.interpolate(x_res1, scale_factor=1.0)
        image_features = self.conv6(x) + F.interpolate(image, scale_factor=1.0)
        msg_features = self.message_processor(message)
        msg_features = msg_features.view(batch_size, 8, 32, 16)
        h, w = image.size(2), image.size(3)
        msg_features = F.interpolate(msg_features, size=(h, w), mode='bilinear', align_corners=False)
        combined = torch.cat([image, msg_features], dim=1)
        stego_image = self.spatial_transformer(combined)
        r_channel = stego_image[:, 0:1, :, :]
        g_channel = stego_image[:, 1:2, :, :]
        b_channel = stego_image[:, 2:3, :, :]
        r_channel = self.r_channel_processor(r_channel)
        g_channel = self.g_channel_processor(g_channel)
        b_channel = self.b_channel_processor(b_channel)
        stego_image = torch.cat([r_channel, g_channel, b_channel], dim=1)
        alpha = 0.01
        stego_image = image + alpha * stego_image
        stego_image = torch.clamp(stego_image, -1, 1)
        return stego_image

class RGBStegoDecoder(nn.Module):
    def __init__(self, message_length=512):
        super(RGBStegoDecoder, self).__init__()
        self.message_length = message_length
        self.r_channel_extractor = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.g_channel_extractor = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.b_channel_extractor = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.combined_processor = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(64 * 16 * 16, 2048)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, message_length)

    def forward(self, stego_image):
        batch_size = stego_image.size(0)
        r_channel = stego_image[:, 0:1, :, :]
        g_channel = stego_image[:, 1:2, :, :]
        b_channel = stego_image[:, 2:3, :, :]
        r_features = F.leaky_relu(self.r_channel_extractor(r_channel), 0.2)
        g_features = F.leaky_relu(self.g_channel_extractor(g_channel), 0.2)
        b_features = F.leaky_relu(self.b_channel_extractor(b_channel), 0.2)
        rgb_features = torch.cat([r_features, g_features, b_features], dim=1)
        rgb_processed = F.leaky_relu(self.combined_processor(rgb_features), 0.2)
        x = F.leaky_relu(self.bn1(self.conv1(stego_image)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2)
        attention = self.channel_attention(x)
        x = x * attention
        x = F.adaptive_avg_pool2d(x, (16, 16))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        message = torch.sigmoid(self.fc3(x))
        return message

# Part 2: Main Steganography System
class HybridStegoSystem:
    def __init__(self, sd_model_id="stabilityai/stable-diffusion-2-1-base", gan_message_length=512):
        self.message_length = gan_message_length
        self.sd_model_id = sd_model_id
        self.sd_pipeline = None
        self.encoder = RGBStegoEncoder().to(device)
        self.decoder = RGBStegoDecoder(message_length=gan_message_length).to(device)
        self.encoder.eval()
        self.decoder.eval()
        print(f"Hybrid steganography system initialized on {device}")

    def _load_sd_pipeline(self):
        if self.sd_pipeline is None:
            print("Loading Stable Diffusion model...")
            self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                self.sd_model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None
            ).to(device)
            print("Stable Diffusion model loaded")

    def load_gan_models(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        self.encoder.eval()
        self.decoder.eval()
        print("GAN models loaded successfully")

    def encode_binary_to_prompt(self, binary_data):
        return encode_binary_to_prompt(binary_data)

    def decode_prompt_to_binary(self, prompt):
        return decode_prompt_to_binary(prompt)

    def hide_data_with_prompt(self, binary_data, seed=None):
        self._load_sd_pipeline()
        prompt = self.encode_binary_to_prompt(binary_data)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        with torch.no_grad():
            image = self.sd_pipeline(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
        return image, prompt

    def hide_data_with_gan(self, cover_image, binary_data, image_size=(256, 256)):
        if isinstance(cover_image, np.ndarray):
            cover_image = torch.from_numpy(cover_image).permute(2, 0, 1).float() / 127.5 - 1.0
        if isinstance(cover_image, Image.Image):
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            cover_image = transform(cover_image)

        if isinstance(binary_data, str):
            message = torch.tensor([int(bit) for bit in binary_data], dtype=torch.float32)
        elif isinstance(binary_data, torch.Tensor):
            message = binary_data
        else:
            raise ValueError("binary_data must be a binary string or tensor")

        message = message[:self.message_length]
        if len(message) > self.message_length:
            print(f"Warning: Message length exceeds {self.message_length} bits. Truncating.")
            message = message[:self.message_length]
        elif len(message) < self.message_length:
            message = torch.cat([message, torch.zeros(self.message_length - len(message))])

        cover_image = cover_image.unsqueeze(0).to(device)
        message = message.unsqueeze(0).to(device)

        with torch.no_grad():
            stego_image = self.encoder(cover_image, message)

        stego_image = (stego_image.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        stego_image = Image.fromarray(stego_image)
        return stego_image

    def extract_data_from_gan(self, stego_image, image_size=(256, 256)):
        if isinstance(stego_image, np.ndarray):
            stego_image = torch.from_numpy(stego_image).permute(2, 0, 1).float() / 127.5 - 1.0
        if isinstance(stego_image, Image.Image):
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            stego_image = transform(stego_image)

        stego_image = stego_image.unsqueeze(0).to(device)
        with torch.no_grad():
            decoded_message = self.decoder(stego_image)
        binary = ''.join(['1' if bit > 0.5 else '0' for bit in decoded_message.squeeze().cpu().numpy()])
        return binary

    def encrypt_and_hide(self, text, key, cover_image=None, method="gan", seed=None):
        cipher = AESCipher(key)
        encrypted_data = cipher.encrypt(text)
        binary_data = ''.join(format(byte, '08b') for byte in encrypted_data)
        if method == "prompt":
            stego_image, prompt = self.hide_data_with_prompt(binary_data, seed=seed)
            return stego_image, prompt
        elif method == "gan":
            if cover_image is None:
                raise ValueError("Cover image is required for GAN-based steganography")
            message_tensor = torch.tensor([int(bit) for bit in binary_data[:self.message_length]], dtype=torch.float32)
            if len(binary_data) > self.message_length:
                print(f"Warning: Message length exceeds {self.message_length} bits. Truncating.")
            if len(message_tensor) < self.message_length:
                message_tensor = torch.cat([
                    message_tensor, 
                    torch.zeros(self.message_length - len(message_tensor))
                ])
            stego_image = self.hide_data_with_gan(cover_image, message_tensor)
            return stego_image
        else:
            raise ValueError("Method must be 'prompt' or 'gan'")

    def encrypt_and_hide_hybrid(self, text, key, cover_image=None, seed=None):
        cipher = AESCipher(key)
        encrypted_data = cipher.encrypt(text)
        binary_data = ''.join(format(byte, '08b') for byte in encrypted_data)
        
        # Split binary data into two parts
        split_point = len(binary_data) // 2
        prompt_binary = binary_data[:split_point]
        gan_binary = binary_data[split_point:]
        
        # Hide first half using prompt-based method
        stego_image, prompt = self.hide_data_with_prompt(prompt_binary, seed=seed)
        
        # Warn if prompt is too long
        token_count = len(prompt.split())
        if token_count > 60:
            print(f"Warning: Prompt has {token_count} tokens, may exceed Stable Diffusion limit (77 tokens).")
        
        # Use the generated image as the cover image for GAN if none provided
        cover_image = stego_image if cover_image is None else cover_image
        
        # Check GAN message length
        if len(gan_binary) > self.message_length:
            print(f"Warning: GAN message length exceeds {self.message_length} bits. Truncating.")
            gan_binary = gan_binary[:self.message_length]
        
        # Convert GAN binary to tensor
        gan_message = torch.tensor([int(bit) for bit in gan_binary], dtype=torch.float32)
        if len(gan_message) < self.message_length:
            gan_message = torch.cat([gan_message, torch.zeros(self.message_length - len(gan_message))])
        
        # Hide second half using GAN-based method
        final_stego_image = self.hide_data_with_gan(cover_image, gan_message)
        
        return final_stego_image, prompt

    def extract_and_decrypt(self, stego_data, key, method="gan"):
        if method == "prompt":
            binary_data = self.decode_prompt_to_binary(stego_data)
        elif method == "gan":
            binary_data = self.extract_data_from_gan(stego_data)
        else:
            raise ValueError("Method must be 'prompt' or 'gan'")
        bytes_data = bytearray()
        for i in range(0, len(binary_data), 8):
            byte = binary_data[i:i+8]
            if len(byte) == 8:
                bytes_data.append(int(byte, 2))
        cipher = AESCipher(key)
        try:
            decrypted_data = cipher.decrypt(bytes(bytes_data))
            text = decrypted_data.decode('utf-8')
            return text
        except Exception as e:
            print(f"Decryption error: {e}")
            return None

    def extract_and_decrypt_hybrid(self, prompt, stego_image, key):
        # Extract first half from prompt
        prompt_binary = self.decode_prompt_to_binary(prompt)
        
        # Extract second half from image
        gan_binary = self.extract_data_from_gan(stego_image)
        
        # Combine binary data
        binary_data = prompt_binary + gan_binary
        
        # Convert to bytes
        bytes_data = bytearray()
        for i in range(0, len(binary_data), 8):
            byte = binary_data[i:i+8]
            if len(byte) == 8:
                bytes_data.append(int(byte, 2))
        
        # Decrypt
        cipher = AESCipher(key)
        try:
            decrypted_data = cipher.decrypt(bytes(bytes_data))
            text = decrypted_data.decode('utf-8')
            return text
        except Exception as e:
            print(f"Decryption error: {e}")
            return None

