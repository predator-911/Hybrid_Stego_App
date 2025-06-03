import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import re
from diffusers import StableDiffusionPipeline
import warnings

# Import improved utils
from stego_utils import (
    AESCipher,
    OBJECT_TABLE,
    COLOR_TABLE,
    STYLE_TABLE,
    MOOD_TABLE,
    INV_OBJECT_TABLE,
    INV_COLOR_TABLE,
    INV_STYLE_TABLE,
    INV_MOOD_TABLE,
    encode_binary_to_prompt,
    decode_prompt_to_binary,
    compress_message,
    validate_prompt_capacity,
    estimate_data_capacity
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RGBStegoEncoder(nn.Module):
    """Improved GAN encoder with better capacity and stability."""
    def __init__(self, message_length=512):
        super(RGBStegoEncoder, self).__init__()
        self.message_length = message_length
        
        # Image processing branch
        self.img_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.img_bn1 = nn.BatchNorm2d(64)
        self.img_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.img_bn2 = nn.BatchNorm2d(128)
        self.img_conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.img_bn3 = nn.BatchNorm2d(128)
        
        # Message processing branch
        self.msg_fc1 = nn.Linear(message_length, 1024)
        self.msg_fc2 = nn.Linear(1024, 2048)
        self.msg_fc3 = nn.Linear(2048, 4096)
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(128 + 16, 64, kernel_size=3, padding=1)
        self.fusion_bn = nn.BatchNorm2d(64)
        
        # Output layers with residual connection
        self.out_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.out_bn1 = nn.BatchNorm2d(32)
        self.out_conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
        # Adaptive scaling
        self.scale_factor = nn.Parameter(torch.tensor(0.1))
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, image, message):
        batch_size, _, h, w = image.shape
        
        # Process image
        x = F.leaky_relu(self.img_bn1(self.img_conv1(image)), 0.2)
        x = F.leaky_relu(self.img_bn2(self.img_conv2(x)), 0.2)
        img_features = F.leaky_relu(self.img_bn3(self.img_conv3(x)), 0.2)
        
        # Process message
        msg = F.leaky_relu(self.msg_fc1(message))
        msg = self.dropout(msg)
        msg = F.leaky_relu(self.msg_fc2(msg))
        msg = self.dropout(msg)
        msg_features = F.leaky_relu(self.msg_fc3(msg))
        
        # Reshape message features to spatial format
        msg_features = msg_features.view(batch_size, 16, 16, 16)
        msg_features = F.interpolate(msg_features, size=(h, w), mode='bilinear', align_corners=False)
        
        # Fuse image and message features
        combined = torch.cat([img_features, msg_features], dim=1)
        fused = F.leaky_relu(self.fusion_bn(self.fusion_conv(combined)), 0.2)
        
        # Generate perturbation
        perturbation = F.leaky_relu(self.out_bn1(self.out_conv1(fused)), 0.2)
        perturbation = torch.tanh(self.out_conv2(perturbation))
        
        # Apply adaptive scaling and residual connection
        stego_image = image + self.scale_factor * perturbation
        stego_image = torch.clamp(stego_image, -1, 1)
        
        return stego_image

class RGBStegoDecoder(nn.Module):
    """Improved GAN decoder with attention mechanism."""
    def __init__(self, message_length=512):
        super(RGBStegoDecoder, self).__init__()
        self.message_length = message_length
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Message reconstruction
        self.global_pool = nn.AdaptiveAvgPool2d(8)
        self.fc1 = nn.Linear(256 * 8 * 8, 2048)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(1024, message_length)

    def forward(self, stego_image):
        batch_size = stego_image.size(0)
        
        # Feature extraction
        x = F.leaky_relu(self.bn1(self.conv1(stego_image)), 0.2)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.max_pool2d(x, 2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        
        # Apply attention mechanisms
        ch_att = self.channel_att(x)
        x = x * ch_att
        
        sp_att = self.spatial_att(x)
        x = x * sp_att
        
        # Global pooling and message reconstruction
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        message = torch.sigmoid(self.fc3(x))
        
        return message

class HybridStegoSystem:
    """Improved hybrid system with better capacity management."""
    def __init__(self, sd_model_id="stabilityai/stable-diffusion-2-1-base", gan_message_length=512):
        self.message_length = gan_message_length
        self.sd_model_id = sd_model_id
        self.sd_pipeline = None
        self.max_prompt_tokens = 60  # Conservative limit
        self.max_prompt_capacity = estimate_data_capacity(self.max_prompt_tokens)
        
        # Initialize GAN models
        self.encoder = RGBStegoEncoder(message_length=gan_message_length).to(device)
        self.decoder = RGBStegoDecoder(message_length=gan_message_length).to(device)
        self.encoder.eval()
        self.decoder.eval()
        
        print(f"Hybrid steganography system initialized on {device}")
        print(f"Prompt capacity: ~{self.max_prompt_capacity} bits")
        print(f"GAN capacity: {gan_message_length} bits")

    def _load_sd_pipeline(self):
        """Load Stable Diffusion pipeline with error handling."""
        if self.sd_pipeline is None:
            try:
                print("Loading Stable Diffusion model...")
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    self.sd_model_id,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                ).to(device)
                
                # Enable memory efficient attention if available
                if hasattr(self.sd_pipeline, 'enable_attention_slicing'):
                    self.sd_pipeline.enable_attention_slicing()
                if hasattr(self.sd_pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.sd_pipeline.enable_xformers_memory_efficient_attention()
                    except:
                        pass
                        
                print("Stable Diffusion model loaded successfully")
            except Exception as e:
                print(f"Error loading Stable Diffusion: {e}")
                raise

    def load_gan_models(self, encoder_path, decoder_path):
        """Load pre-trained GAN models."""
        try:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
            self.encoder.eval()
            self.decoder.eval()
            print("GAN models loaded successfully")
        except Exception as e:
            print(f"Error loading GAN models: {e}")
            raise

    def _prepare_message_for_method(self, text, method):
        """Prepare message based on steganography method."""
        if method == "prompt":
            # Check if message fits in prompt capacity
            estimated_bits = len(text.encode('utf-8')) * 8
            if estimated_bits > self.max_prompt_capacity:
                compressed_text = compress_message(text, self.max_prompt_capacity)
                print(f"Message compressed from {len(text)} to {len(compressed_text)} characters")
                return compressed_text
            return text
        elif method == "gan":
            # Check if message fits in GAN capacity
            estimated_bits = len(text.encode('utf-8')) * 8
            if estimated_bits > self.message_length:
                max_chars = self.message_length // 8
                compressed_text = compress_message(text, self.message_length)
                print(f"Message truncated to fit GAN capacity ({max_chars} characters)")
                return compressed_text
            return text
        return text

    def encrypt_and_hide(self, text, key, cover_image=None, method="gan", seed=None):
        """Main encryption and hiding function with improved error handling."""
        try:
            # Prepare message
            text = self._prepare_message_for_method(text, method)
            
            # Encrypt
            cipher = AESCipher(key)
            encrypted_data = cipher.encrypt(text)
            binary_data = ''.join(format(byte, '08b') for byte in encrypted_data)
            
            if method == "prompt":
                return self._hide_with_prompt(binary_data, seed)
            elif method == "gan":
                if cover_image is None:
                    raise ValueError("Cover image is required for GAN-based steganography")
                return self._hide_with_gan(cover_image, binary_data)
            else:
                raise ValueError("Method must be 'prompt' or 'gan'")
                
        except Exception as e:
            print(f"Error in encrypt_and_hide: {e}")
            raise

    def _hide_with_prompt(self, binary_data, seed=None):
        """Hide data using prompt-based method."""
        self._load_sd_pipeline()
        
        # Generate prompt with token limit consideration
        prompt = encode_binary_to_prompt(binary_data, max_tokens=self.max_prompt_tokens)
        
        # Validate prompt
        is_valid, token_count = validate_prompt_capacity(prompt, max_tokens=77)
        if not is_valid:
            print(f"Warning: Generated prompt has {token_count} tokens (>77). Image generation may fail.")
        
        # Generate image
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        
        with torch.no_grad():
            try:
                result = self.sd_pipeline(
                    prompt,
                    num_inference_steps=20,  # Reduced for speed
                    guidance_scale=7.5,
                    generator=generator,
                    height=512,
                    width=512
                )
                image = result.images[0]
                
                return image, prompt
                
            except Exception as e:
                print(f"Image generation failed: {e}")
                # Fallback to simpler prompt
                simple_prompt = "a red cat, digital art"
                result = self.sd_pipeline(
                    simple_prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    generator=generator
                )
                return result.images[0], prompt

    def _hide_with_gan(self, cover_image, binary_data, image_size=(256, 256)):
        """Hide data using GAN-based method."""
        # Prepare cover image
        if isinstance(cover_image, Image.Image):
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            cover_tensor = transform(cover_image).unsqueeze(0).to(device)
        else:
            raise ValueError("Cover image must be PIL Image")
        
        # Prepare message
        if isinstance(binary_data, str):
            message = torch.tensor([float(bit) for bit in binary_data], dtype=torch.float32)
        else:
            message = binary_data.float()
        
        # Pad or truncate message
        if len(message) > self.message_length:
            message = message[:self.message_length]
        elif len(message) < self.message_length:
            padding = torch.zeros(self.message_length - len(message))
            message = torch.cat([message, padding])
        
        message = message.unsqueeze(0).to(device)
        
        # Generate stego image
        with torch.no_grad():
            stego_tensor = self.encoder(cover_tensor, message)
        
        # Convert back to PIL Image
        stego_image = (stego_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
        stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(stego_image)

    def encrypt_and_hide_hybrid(self, text, key, cover_image=None, seed=None):
        """Hybrid method with intelligent data splitting."""
        try:
            cipher = AESCipher(key)
            encrypted_data = cipher.encrypt(text)
            binary_data = ''.join(format(byte, '08b') for byte in encrypted_data)
            
            # Calculate optimal split based on capacities
            prompt_capacity = self.max_prompt_capacity
            gan_capacity = self.message_length
            total_capacity = prompt_capacity + gan_capacity
            
            if len(binary_data) > total_capacity:
                print(f"Warning: Data exceeds hybrid capacity. Truncating from {len(binary_data)} to {total_capacity} bits")
                binary_data = binary_data[:total_capacity]
            
            # Smart splitting: more critical data in prompt (more reliable)
            prompt_data_len = min(len(binary_data) // 2, prompt_capacity)
            prompt_binary = binary_data[:prompt_data_len]
            gan_binary = binary_data[prompt_data_len:]
            
            # Hide first part in prompt
            stego_image, prompt = self._hide_with_prompt(prompt_binary, seed)
            
            # Use generated image as cover if none provided
            final_cover = cover_image if cover_image else stego_image
            
            # Hide second part in image
            if gan_binary:
                final_stego_image = self._hide_with_gan(final_cover, gan_binary)
            else:
                final_stego_image = stego_image
            
            return final_stego_image, prompt
            
        except Exception as e:
            print(f"Error in hybrid encryption: {e}")
            raise

    def extract_and_decrypt(self, stego_data, key, method="gan", prompt=None):
        """Extract and decrypt hidden message."""
        try:
            if method == "prompt":
                if prompt is None:
                    raise ValueError("Prompt is required for prompt-based extraction")
                return self._extract_from_prompt(prompt, key)
            elif method == "gan":
                if not isinstance(stego_data, Image.Image):
                    raise ValueError("Stego data must be PIL Image for GAN extraction")
                return self._extract_from_gan(stego_data, key)
            else:
                raise ValueError("Method must be 'prompt' or 'gan'")
                
        except Exception as e:
            print(f"Error in extract_and_decrypt: {e}")
            raise

    def _extract_from_prompt(self, prompt, key):
        """Extract data from prompt-based steganography."""
        try:
            # Decode binary data from prompt
            binary_data = decode_prompt_to_binary(prompt)
            
            if not binary_data:
                raise ValueError("No data found in prompt")
            
            # Convert binary to bytes
            byte_data = []
            for i in range(0, len(binary_data), 8):
                if i + 8 <= len(binary_data):
                    byte_value = int(binary_data[i:i+8], 2)
                    byte_data.append(byte_value)
            
            encrypted_data = bytes(byte_data)
            
            # Decrypt
            cipher = AESCipher(key)
            decrypted_text = cipher.decrypt(encrypted_data)
            
            return decrypted_text
            
        except Exception as e:
            print(f"Error extracting from prompt: {e}")
            raise

    def _extract_from_gan(self, stego_image, key, image_size=(256, 256)):
        """Extract data from GAN-based steganography."""
        try:
            # Prepare stego image
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            stego_tensor = transform(stego_image).unsqueeze(0).to(device)
            
            # Extract message
            with torch.no_grad():
                message_tensor = self.decoder(stego_tensor)
            
            # Convert to binary
            message_bits = (message_tensor.squeeze().cpu().numpy() > 0.5).astype(int)
            binary_data = ''.join(str(bit) for bit in message_bits)
            
            # Remove padding (find last meaningful bit)
            last_byte_idx = len(binary_data) - (len(binary_data) % 8)
            binary_data = binary_data[:last_byte_idx]
            
            # Convert to bytes
            byte_data = []
            for i in range(0, len(binary_data), 8):
                if i + 8 <= len(binary_data):
                    byte_value = int(binary_data[i:i+8], 2)
                    byte_data.append(byte_value)
            
            encrypted_data = bytes(byte_data)
            
            # Decrypt
            cipher = AESCipher(key)
            decrypted_text = cipher.decrypt(encrypted_data)
            
            return decrypted_text
            
        except Exception as e:
            print(f"Error extracting from GAN: {e}")
            raise

    def extract_and_decrypt_hybrid(self, stego_image, key, prompt):
        """Extract and decrypt from hybrid steganography."""
        try:
            # Extract from prompt (first part)
            prompt_text = self._extract_from_prompt(prompt, key + "_prompt")
            
            # Extract from GAN (second part) 
            gan_text = self._extract_from_gan(stego_image, key + "_gan")
            
            # Combine parts (implementation depends on how data was split)
            combined_text = prompt_text + gan_text
            
            return combined_text
            
        except Exception as e:
            print(f"Error in hybrid extraction: {e}")
            # Try individual methods as fallback
            try:
                return self._extract_from_prompt(prompt, key)
            except:
                try:
                    return self._extract_from_gan(stego_image, key)
                except:
                    raise e

    def calculate_capacity(self):
        """Calculate total system capacity."""
        return {
            'prompt_capacity_bits': self.max_prompt_capacity,
            'gan_capacity_bits': self.message_length,
            'hybrid_capacity_bits': self.max_prompt_capacity + self.message_length,
            'prompt_capacity_chars': self.max_prompt_capacity // 8,
            'gan_capacity_chars': self.message_length // 8,
            'hybrid_capacity_chars': (self.max_prompt_capacity + self.message_length) // 8
        }

    def benchmark_methods(self, test_text, key, cover_image=None, seed=42):
        """Benchmark different steganography methods."""
        results = {}
        
        try:
            # Test prompt method
            print("Testing prompt-based method...")
            start_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            
            if start_time:
                start_time.record()
            
            stego_img, prompt = self.encrypt_and_hide(test_text, key, method="prompt", seed=seed)
            extracted_text = self.extract_and_decrypt(None, key, method="prompt", prompt=prompt)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                prompt_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                prompt_time = 0
                
            results['prompt'] = {
                'success': extracted_text == test_text,
                'time_seconds': prompt_time,
                'capacity_used': len(test_text.encode('utf-8')) * 8
            }
            
        except Exception as e:
            results['prompt'] = {'success': False, 'error': str(e)}
        
        try:
            # Test GAN method
            if cover_image:
                print("Testing GAN-based method...")
                if start_time:
                    start_time.record()
                
                stego_img = self.encrypt_and_hide(test_text, key, cover_image, method="gan")
                extracted_text = self.extract_and_decrypt(stego_img, key, method="gan")
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    gan_time = start_time.elapsed_time(end_time) / 1000.0
                else:
                    gan_time = 0
                    
                results['gan'] = {
                    'success': extracted_text == test_text,
                    'time_seconds': gan_time,
                    'capacity_used': len(test_text.encode('utf-8')) * 8
                }
            else:
                results['gan'] = {'success': False, 'error': 'No cover image provided'}
                
        except Exception as e:
            results['gan'] = {'success': False, 'error': str(e)}
        
        return results

    def save_models(self, encoder_path, decoder_path):
        """Save trained GAN models."""
        try:
            torch.save(self.encoder.state_dict(), encoder_path)
            torch.save(self.decoder.state_dict(), decoder_path)
            print(f"Models saved to {encoder_path} and {decoder_path}")
        except Exception as e:
            print(f"Error saving models: {e}")
            raise
