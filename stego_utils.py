import re
import torch
import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# Encoding tables for prompt-based steganography
OBJECT_TABLE = {
    0: "house", 1: "tree", 2: "car", 3: "person", 4: "bird", 5: "mountain",
    6: "river", 7: "dog", 8: "cat", 9: "flower", 10: "city", 11: "ocean",
    12: "forest", 13: "bridge", 14: "lake", 15: "rocket", 16: "castle",
    17: "boat", 18: "moon", 19: "sun", 20: "cloud", 21: "butterfly",
    22: "horse", 23: "statue", 24: "waterfall", 25: "island", 26: "desert",
    27: "galaxy", 28: "robot", 29: "dragon", 30: "unicorn", 31: "phoenix"
}

COLOR_TABLE = {
    0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "purple",
    5: "orange", 6: "teal", 7: "silver", 8: "gold", 9: "copper",
    10: "emerald", 11: "sapphire", 12: "ruby", 13: "amber", 14: "ivory",
    15: "crimson"
}

STYLE_TABLE = {
    0: "impressionist", 1: "minimalist", 2: "surrealist", 3: "abstract",
    4: "photorealistic", 5: "cubist", 6: "pop art", 7: "watercolor",
    8: "digital art", 9: "oil painting", 10: "sketch", 11: "anime",
    12: "vaporwave", 13: "synthwave", 14: "cyberpunk", 15: "renaissance"
}

HIGHLIGHT_TABLE = {
    0: "glowing", 1: "subtle", 2: "dramatic", 3: "lens flare",
    4: "backlit", 5: "neon", 6: "high contrast", 7: "soft",
    8: "cinematic", 9: "ethereal", 10: "mystical", 11: "volumetric",
    12: "studio", 13: "natural", 14: "vibrant", 15: "monochromatic"
}

INV_OBJECT_TABLE = {v: k for k, v in OBJECT_TABLE.items()}
INV_COLOR_TABLE = {v: k for k, v in COLOR_TABLE.items()}
INV_STYLE_TABLE = {v: k for k, v in STYLE_TABLE.items()}
INV_HIGHLIGHT_TABLE = {v: k for k, v in HIGHLIGHT_TABLE.items()}

class AESCipher:
    def __init__(self, key=None):
        if key is None:
            self.key = get_random_bytes(32)
        else:
            if isinstance(key, str):
                key_bytes = key.encode('utf-8')
                if len(key_bytes) != 32:
                    print("Warning: Key length is not 32 bytes. Hashing with SHA-256.")
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

def binary_to_chunks(binary_data, chunk_size=17):
    return [binary_data[i:i+chunk_size] for i in range(0, len(binary_data), chunk_size)]

def encode_binary_to_prompt(binary_data):
    if isinstance(binary_data, bytes):
        binary_data = ''.join(format(byte, '08b') for byte in binary_data)
    elif isinstance(binary_data, str) and all(bit in '01' for bit in binary_data):
        pass
    else:
        raise ValueError("binary_data must be bytes or a string of 0s and 1s")
    chunks = binary_to_chunks(binary_data)
    prompt_parts = []
    for chunk in chunks:
        object_bits = chunk[:5] if len(chunk) >= 5 else chunk.ljust(5, '0')
        color_bits = chunk[5:9] if len(chunk) >= 9 else chunk[5:].ljust(4, '0')
        style_bits = chunk[9:13] if len(chunk) >= 13 else chunk[9:].ljust(4, '0')
        highlight_bits = chunk[13:17] if len(chunk) >= 17 else chunk[13:].ljust(4, '0')
        object_index = min(int(object_bits, 2), len(OBJECT_TABLE) - 1)
        color_index = min(int(color_bits, 2), len(COLOR_TABLE) - 1)
        style_index = min(int(style_bits, 2), len(STYLE_TABLE) - 1)
        highlight_index = min(int(highlight_bits, 2), len(HIGHLIGHT_TABLE) - 1)
        object_word = OBJECT_TABLE[object_index]
        color_word = COLOR_TABLE[color_index]
        style_word = STYLE_TABLE[style_index]
        highlight_word = HIGHLIGHT_TABLE[highlight_index]
        prompt_part = f"a {object_word} in {color_word} tones, {style_word} style, with {highlight_word} effects"
        prompt_parts.append(prompt_part)
    separator = ", and " if len(prompt_parts) > 1 else ""
    final_prompt = "A painting of " + separator.join(prompt_parts) + "."
    token_count = len(final_prompt.split())
    if token_count > 60:
        print(f"Warning: Prompt has {token_count} tokens, may exceed Stable Diffusion limit (77 tokens).")
    return final_prompt, token_count

def decode_prompt_to_binary(prompt):
    binary_data = ""
    object_pattern = r"(?:a|an) ({})\b".format("|".join(INV_OBJECT_TABLE.keys()))
    color_pattern = r"in ({})\b".format("|".join(INV_COLOR_TABLE.keys()))
    style_pattern = r"({})\s+style".format("|".join(INV_STYLE_TABLE.keys()))
    highlight_pattern = r"with ({})\b".format("|".join(INV_HIGHLIGHT_TABLE.keys()))
    objects = re.findall(object_pattern, prompt.lower())
    colors = re.findall(color_pattern, prompt.lower())
    styles = re.findall(style_pattern, prompt.lower())
    highlights = re.findall(highlight_pattern, prompt.lower())
    min_elements = min(len(objects), len(colors), len(styles), len(highlights))
    if min_elements == 0:
        raise ValueError("No valid semantic elements found in prompt")
    for i in range(min_elements):
        try:
            object_bits = format(INV_OBJECT_TABLE.get(objects[i], 0), '05b')
            color_bits = format(INV_COLOR_TABLE.get(colors[i], 0), '04b')
            style_bits = format(INV_STYLE_TABLE.get(styles[i], 0), '04b')
            highlight_bits = format(INV_HIGHLIGHT_TABLE.get(highlights[i], 0), '04b')
            chunk_bits = object_bits + color_bits + style_bits + highlight_bits
            binary_data += chunk_bits
        except (KeyError, IndexError) as e:
            raise ValueError(f"Error decoding prompt element: {e}")
    return binary_data

