import re
import torch
import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Optimized encoding tables for better token efficiency
OBJECT_TABLE = {
    0: "cat", 1: "dog", 2: "bird", 3: "fish", 4: "tree", 5: "flower",
    6: "house", 7: "car", 8: "boat", 9: "moon", 10: "sun", 11: "star",
    12: "cloud", 13: "lake", 14: "hill", 15: "cave", 16: "fox", 17: "deer",
    18: "owl", 19: "rose", 20: "oak", 21: "pine", 22: "barn", 23: "truck",
    24: "ship", 25: "fire", 26: "snow", 27: "rock", 28: "field", 29: "path",
    30: "bridge", 31: "tower"
}

COLOR_TABLE = {
    0: "red", 1: "blue", 2: "green", 3: "gold", 4: "silver", 5: "black",
    6: "white", 7: "pink", 8: "purple", 9: "orange", 10: "brown", 11: "gray",
    12: "yellow", 13: "teal", 14: "coral", 15: "jade"
}

STYLE_TABLE = {
    0: "oil", 1: "sketch", 2: "digital", 3: "photo", 4: "anime", 5: "pixel",
    6: "water", 7: "ink", 8: "pencil", 9: "chalk", 10: "pastel", 11: "neon",
    12: "retro", 13: "modern", 14: "vintage", 15: "abstract"
}

MOOD_TABLE = {
    0: "calm", 1: "bright", 2: "dark", 3: "warm", 4: "cool", 5: "soft",
    6: "sharp", 7: "misty", 8: "clear", 9: "foggy", 10: "sunny", 11: "stormy",
    12: "peaceful", 13: "dramatic", 14: "serene", 15: "vibrant"
}

# Inverse mappings
INV_OBJECT_TABLE = {v: k for k, v in OBJECT_TABLE.items()}
INV_COLOR_TABLE = {v: k for k, v in COLOR_TABLE.items()}
INV_STYLE_TABLE = {v: k for k, v in STYLE_TABLE.items()}
INV_MOOD_TABLE = {v: k for k, v in MOOD_TABLE.items()}

class AESCipher:
    def __init__(self, key=None):
        if key is None:
            self.key = get_random_bytes(32)
        else:
            if isinstance(key, str):
                key_bytes = key.encode('utf-8')
                if len(key_bytes) != 32:
                    self.key = hashlib.sha256(key_bytes).digest()
                else:
                    self.key = key_bytes
            else:
                self.key = key

    def encrypt(self, plaintext):
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        iv = get_random_bytes(16)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        padded_data = pad(plaintext, AES.block_size)
        ciphertext = cipher.encrypt(padded_data)
        return iv + ciphertext

    def decrypt(self, ciphertext):
        iv = ciphertext[:16]
        ciphertext = ciphertext[16:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        decrypted = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted

    def get_key_hex(self):
        return self.key.hex()

    @classmethod
    def from_hex_key(cls, hex_key):
        key = bytes.fromhex(hex_key)
        return cls(key)

def encode_binary_to_prompt(binary_data, max_tokens=60):
    """
    Optimized encoding that respects token limits.
    Uses 20-bit chunks (5+4+4+4+3 bits) with padding optimization.
    """
    if isinstance(binary_data, bytes):
        binary_data = ''.join(format(byte, '08b') for byte in binary_data)
    elif not (isinstance(binary_data, str) and all(bit in '01' for bit in binary_data)):
        raise ValueError("binary_data must be bytes or a string of 0s and 1s")
    
    # Calculate maximum data we can encode within token limit
    # Each chunk creates ~6-8 tokens, so for 60 tokens max, we can have ~8-10 chunks max
    max_chunks = max_tokens // 7  # Conservative estimate
    max_bits = max_chunks * 20
    
    if len(binary_data) > max_bits:
        print(f"Warning: Data length ({len(binary_data)} bits) exceeds capacity ({max_bits} bits). Truncating.")
        binary_data = binary_data[:max_bits]
    
    # Pad to multiple of 20 bits
    padding_needed = (20 - len(binary_data) % 20) % 20
    binary_data += '0' * padding_needed
    
    # Split into 20-bit chunks
    chunks = [binary_data[i:i+20] for i in range(0, len(binary_data), 20)]
    
    prompt_parts = []
    for chunk in chunks:
        if len(chunk) < 20:
            chunk = chunk.ljust(20, '0')
            
        object_bits = chunk[:5]    # 32 objects
        color_bits = chunk[5:9]    # 16 colors  
        style_bits = chunk[9:13]   # 16 styles
        mood_bits = chunk[13:17]   # 16 moods
        extra_bits = chunk[17:20]  # 3 extra bits for future use
        
        object_idx = int(object_bits, 2) % len(OBJECT_TABLE)
        color_idx = int(color_bits, 2) % len(COLOR_TABLE)
        style_idx = int(style_bits, 2) % len(STYLE_TABLE)
        mood_idx = int(mood_bits, 2) % len(MOOD_TABLE)
        
        # Create more natural, shorter phrases
        if len(prompt_parts) == 0:
            prompt_part = f"{color_idx % 2 and 'a' or 'the'} {COLOR_TABLE[color_idx]} {OBJECT_TABLE[object_idx]}"
        else:
            prompt_part = f"{COLOR_TABLE[color_idx]} {OBJECT_TABLE[object_idx]}"
            
        prompt_parts.append(prompt_part)
    
    # Create compact prompt
    if len(prompt_parts) == 1:
        base_prompt = prompt_parts[0]
    elif len(prompt_parts) <= 3:
        base_prompt = ", ".join(prompt_parts[:-1]) + " and " + prompt_parts[-1]
    else:
        base_prompt = ", ".join(prompt_parts[:3]) + " and more"
    
    # Add style and mood from first chunk
    if chunks:
        first_chunk = chunks[0]
        style_idx = int(first_chunk[9:13], 2) % len(STYLE_TABLE)
        mood_idx = int(first_chunk[13:17], 2) % len(MOOD_TABLE)
        final_prompt = f"{base_prompt}, {STYLE_TABLE[style_idx]} style, {MOOD_TABLE[mood_idx]} mood"
    else:
        final_prompt = base_prompt
    
    # Verify token count
    token_count = len(final_prompt.split())
    if token_count > max_tokens:
        # Fallback to minimal prompt
        if chunks:
            first_chunk = chunks[0]
            object_idx = int(first_chunk[:5], 2) % len(OBJECT_TABLE)
            color_idx = int(first_chunk[5:9], 2) % len(COLOR_TABLE)
            final_prompt = f"a {COLOR_TABLE[color_idx]} {OBJECT_TABLE[object_idx]}"
        else:
            final_prompt = "a red cat"
    
    return final_prompt

def decode_prompt_to_binary(prompt):
    """
    Improved decoding with better error handling and recovery.
    """
    binary_data = ""
    
    # Normalize prompt
    prompt_lower = prompt.lower()
    
    # Extract objects
    object_matches = []
    for word, idx in INV_OBJECT_TABLE.items():
        if word in prompt_lower:
            object_matches.append((idx, prompt_lower.find(word)))
    object_matches.sort(key=lambda x: x[1])  # Sort by position
    
    # Extract colors
    color_matches = []
    for word, idx in INV_COLOR_TABLE.items():
        if word in prompt_lower:
            color_matches.append((idx, prompt_lower.find(word)))
    color_matches.sort(key=lambda x: x[1])
    
    # Extract styles
    style_matches = []
    for word, idx in INV_STYLE_TABLE.items():
        if word in prompt_lower:
            style_matches.append((idx, prompt_lower.find(word)))
    
    # Extract moods
    mood_matches = []
    for word, idx in INV_MOOD_TABLE.items():
        if word in prompt_lower:
            mood_matches.append((idx, prompt_lower.find(word)))
    
    if not object_matches and not color_matches:
        raise ValueError("No recognizable semantic elements found in prompt")
    
    # Reconstruct chunks
    max_items = max(len(object_matches), len(color_matches), 1)
    
    for i in range(max_items):
        # Get indices with fallbacks
        obj_idx = object_matches[i][0] if i < len(object_matches) else 0
        color_idx = color_matches[i][0] if i < len(color_matches) else 0
        style_idx = style_matches[0][0] if style_matches else 0
        mood_idx = mood_matches[0][0] if mood_matches else 0
        
        # Convert to binary
        obj_bits = format(obj_idx, '05b')
        color_bits = format(color_idx, '04b')
        style_bits = format(style_idx, '04b')
        mood_bits = format(mood_idx, '04b')
        extra_bits = '000'  # Padding
        
        chunk_bits = obj_bits + color_bits + style_bits + mood_bits + extra_bits
        binary_data += chunk_bits
    
    return binary_data

def compress_message(message, max_capacity_bits):
    """
    Simple compression for messages that exceed capacity.
    """
    if len(message) * 8 <= max_capacity_bits:
        return message
    
    # Simple truncation with ellipsis
    max_chars = (max_capacity_bits // 8) - 3  # Reserve space for "..."
    if max_chars > 0:
        return message[:max_chars] + "..."
    else:
        return message[:max_capacity_bits // 8]

# Additional utility functions for better error handling
def validate_prompt_capacity(prompt, max_tokens=77):
    """Validate if prompt fits within token limits."""
    token_count = len(prompt.split())
    return token_count <= max_tokens, token_count

def estimate_data_capacity(max_tokens=60):
    """Estimate maximum data capacity for given token limit."""
    max_chunks = max_tokens // 7
    return max_chunks * 20  # 20 bits per chunk
