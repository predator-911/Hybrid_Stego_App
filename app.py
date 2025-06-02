import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import time
from stego_utils import AESCipher, encode_binary_to_prompt, decode_prompt_to_binary
from stego_models import RGBStegoEncoder, RGBStegoDecoder, HybridStegoSystem

# Set page configuration
st.set_page_config(
    page_title="Hybrid Steganography System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to validate key strength
def validate_key(key):
    if len(key) < 8:
        return False, "Warning: Encryption key is too short (< 8 characters). Use a stronger key for better security."
    return True, ""

# Function to load models
@st.cache_resource
def load_stego_system():
    system = HybridStegoSystem(
        sd_model_id="stabilityai/stable-diffusion-2-1-base",
        gan_message_length=512
    )
    
    encoder_path = "stego_gan_encoder.pth"
    decoder_path = "stego_gan_decoder.pth"
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        try:
            system.load_gan_models(encoder_path, decoder_path)
            st.sidebar.success("✅ Pre-trained GAN models loaded successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Could not load pre-trained models: {e}")
    else:
        st.sidebar.warning("⚠️ Pre-trained GAN models not found. Using untrained models.")
        
    return system

# Initialize the steganography system
stego_system = load_stego_system()

# Create sidebar with information
with st.sidebar:
    st.title("🔐 Hybrid Steganography")
    st.write("""
    This app demonstrates three approaches to steganography:
    
    1. **Prompt-based**: Hides data in text prompts used for image generation
    2. **GAN-based**: Embeds encrypted data in image pixels
    3. **Hybrid Prompt-GAN**: Splits the message, hiding half in the prompt and half in the image
    """)
    
    st.write("---")
    st.subheader("How It Works")
    st.markdown("""
    - **Prompt Steganography**: Encodes binary data into text prompts for Stable Diffusion.
    - **GAN Steganography**: Uses neural networks to hide data in image RGB channels.
    - **Hybrid Steganography**: Combines both for enhanced security and capacity.
    """)
    
    st.write("---")
    st.write(f"🖥️ Running on: **{device}**")
    st.caption("v1.0.0 | Hybrid Steganography System")

# Main app title
st.title("Hybrid Prompt-GAN Steganography System")
st.markdown("Hide encrypted messages using state-of-the-art steganography techniques")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Prompt Steganography", "GAN Steganography", "Hybrid Prompt-GAN", "About"])

# Tab 1: Prompt-based Steganography
with tab1:
    st.header("Prompt-based Steganography")
    st.write("Hide your message in a text prompt used to generate an image with Stable Diffusion.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        secret_message = st.text_area(
            "Enter your secret message:",
            "This is a secret message hidden in a prompt-generated image!",
            height=100,
            key="prompt_message"
        )
        
        encryption_key = st.text_input(
            "Enter encryption key:",
            "MySecretKey123",
            type="password",
            key="prompt_key"
        )
        key_valid, key_warning = validate_key(encryption_key)
        if key_warning:
            st.warning(key_warning)
        
        use_seed = st.checkbox("Use fixed seed for reproducibility", value=True, key="prompt_seed")
        seed = st.number_input("Seed value:", value=42, disabled=not use_seed, key="prompt_seed_value")
        
        if st.button("Generate Steganographic Image", key="generate_prompt"):
            if not secret_message:
                st.warning("Please enter a secret message.")
            elif not encryption_key:
                st.warning("Please enter an encryption key.")
            else:
                with st.spinner("Generating image with hidden message..."):
                    try:
                        start_time = time.time()
                        stego_image, prompt = stego_system.encrypt_and_hide(
                            secret_message,
                            encryption_key,
                            method="prompt",
                            seed=seed if use_seed else None
                        )
                        token_count = len(prompt.split())
                        if token_count > 60:
                            st.warning(f"⚠️ Prompt has {token_count} tokens, may exceed Stable Diffusion's limit (77 tokens).")
                        elapsed_time = time.time() - start_time
                        st.session_state.prompt_stego_image = stego_image
                        st.session_state.prompt_used = prompt
                        st.success(f"✅ Image generated in {elapsed_time:.2f} seconds!")
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
    
    with col2:
        st.subheader("Output")
        if 'prompt_stego_image' in st.session_state:
            st.image(
                st.session_state.prompt_stego_image,
                caption="Generated image with hidden message",
                use_column_width=True
            )
            buf = io.BytesIO()
            st.session_state.prompt_stego_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="stego_image_prompt.png",
                mime="image/png",
                key="download_prompt_image"
            )
            with st.expander("View Generated Prompt"):
                st.code(st.session_state.prompt_used)
            st.button(
                "Copy Prompt to Clipboard",
                help="Copy the prompt to share with the recipient",
                key="copy_prompt"
            )
    
    st.markdown("---")
    st.subheader("Extract Message from Prompt")
    extraction_prompt = st.text_area(
        "Enter the prompt containing the hidden message:",
        height=150,
        key="extract_prompt"
    )
    extraction_key = st.text_input(
        "Enter the decryption key:",
        type="password",
        key="extract_key_prompt"
    )
    key_valid, key_warning = validate_key(extraction_key)
    if key_warning:
        st.warning(key_warning)
    
    if st.button("Extract Message", key="extract_prompt_btn"):
        if not extraction_prompt:
            st.warning("Please enter a prompt to extract from.")
        elif not extraction_key:
            st.warning("Please enter a decryption key.")
        else:
            try:
                with st.spinner("Extracting hidden message..."):
                    extracted_message = stego_system.extract_and_decrypt(
                        extraction_prompt,
                        extraction_key,
                        method="prompt"
                    )
                    if extracted_message:
                        st.success("Message extracted successfully!")
                        st.info(f"**Extracted message:** {extracted_message}")
                    else:
                        st.error("Failed to extract message. Check your prompt and decryption key.")
            except Exception as e:
                st.error(f"Error during extraction: {e}")

# Tab 2: GAN-based Steganography
with tab2:
    st.header("GAN-based Steganography")
    st.write("Hide your encrypted message directly in the pixels of an image.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        cover_image = st.file_uploader(
            "Upload cover image (PNG, JPG):",
            type=["png", "jpg", "jpeg"],
            help="This is the image where your message will be hidden.",
            key="gan_cover_image"
        )
        
        if cover_image:
            try:
                cover_img = Image.open(cover_image).convert("RGB")
                if max(cover_img.size) > 2048:
                    st.error("Image is too large. Please upload an image smaller than 2048x2048 pixels.")
                else:
                    if max(cover_img.size) > 512:
                        cover_img = cover_img.resize(
                            (512, int(512 * cover_img.size[1] / cover_img.size[0]))
                            if cover_img.size[0] > cover_img.size[1]
                            else (int(512 * cover_img.size[0] / cover_img.size[1]), 512)
                        )
                    st.image(cover_img, caption="Cover Image", use_column_width=True)
                    st.session_state.cover_image = cover_img
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        gan_secret_message = st.text_area(
            "Enter your secret message:",
            "This is a secret message hidden in an image using GAN steganography!",
            height=100,
            key="gan_message"
        )
        message_length = len(gan_secret_message.encode('utf-8')) * 8
        if message_length > 512:
            st.warning(f"⚠️ Message length ({message_length} bits) exceeds GAN capacity (512 bits). It will be truncated.")
        
        gan_encryption_key = st.text_input(
            "Enter encryption key:",
            "MySecretKey123",
            type="password",
            key="gan_key"
        )
        key_valid, key_warning = validate_key(gan_encryption_key)
        if key_warning:
            st.warning(key_warning)
        
        if st.button("Hide Message in Image", key="generate_gan"):
            if not cover_image:
                st.warning("Please upload a cover image.")
            elif not gan_secret_message:
                st.warning("Please enter a secret message.")
            elif not gan_encryption_key:
                st.warning("Please enter an encryption key.")
            else:
                try:
                    with st.spinner("Hiding message in image..."):
                        start_time = time.time()
                        stego_image = stego_system.encrypt_and_hide(
                            gan_secret_message,
                            gan_encryption_key,
                            st.session_state.cover_image,
                            method="gan"
                        )
                        elapsed_time = time.time() - start_time
                        st.session_state.gan_stego_image = stego_image
                        st.success(f"✅ Message hidden in image in {elapsed_time:.2f} seconds!")
                except Exception as e:
                    st.error(f"Error hiding message: {e}")
    
    with col2:
        st.subheader("Output")
        if 'gan_stego_image' in st.session_state:
            st.image(
                st.session_state.gan_stego_image,
                caption="Image with hidden message",
                use_column_width=True
            )
            buf = io.BytesIO()
            st.session_state.gan_stego_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Stego Image",
                data=byte_im,
                file_name="stego_image_gan.png",
                mime="image/png",
                key="download_gan_image"
            )
            if 'cover_image' in st.session_state:
                with st.expander("Compare Original vs Stego Image"):
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        st.image(st.session_state.cover_image, caption="Original Image")
                    with comp_col2:
                        st.image(st.session_state.gan_stego_image, caption="Stego Image")
    
    st.markdown("---")
    st.subheader("Extract Message from Image")
    stego_image_upload = st.file_uploader(
        "Upload stego image with hidden message:",
        type=["png", "jpg", "jpeg"],
        key="extract_image_upload"
    )
    
    if stego_image_upload:
        try:
            extract_stego_img = Image.open(stego_image_upload).convert("RGB")
            if max(extract_stego_img.size) > 2048:
                st.error("Image is too large. Please upload an image smaller than 2048x2048 pixels.")
            else:
                if max(extract_stego_img.size) > 512:
                    extract_stego_img = extract_stego_img.resize(
                        (512, int(512 * extract_stego_img.size[1] / extract_stego_img.size[0]))
                        if extract_stego_img.size[0] > extract_stego_img.size[1]
                        else (int(512 * extract_stego_img.size[0] / extract_stego_img.size[1]), 512)
                    )
                st.image(extract_stego_img, caption="Stego Image", width=300)
                st.session_state.extract_stego_image = extract_stego_img
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    extraction_gan_key = st.text_input(
        "Enter the decryption key:",
        type="password",
        key="extract_key_gan"
    )
    key_valid, key_warning = validate_key(extraction_gan_key)
    if key_warning:
        st.warning(key_warning)
    
    if st.button("Extract Message", key="extract_gan_btn"):
        if not stego_image_upload:
            st.warning("Please upload a stego image to extract from.")
        elif not extraction_gan_key:
            st.warning("Please enter a decryption key.")
        else:
            try:
                with st.spinner("Extracting hidden message..."):
                    extracted_message = stego_system.extract_and_decrypt(
                        st.session_state.extract_stego_image,
                        extraction_gan_key,
                        method="gan"
                    )
                    if extracted_message:
                        st.success("Message extracted successfully!")
                        st.info(f"**Extracted message:** {extracted_message}")
                    else:
                        st.error("Failed to extract message. Check your image and decryption key.")
            except Exception as e:
                st.error(f"Error during extraction: {e}")

# Tab 3: Hybrid Prompt-GAN Steganography
with tab3:
    st.header("Hybrid Prompt-GAN Steganography")
    st.write("This method splits your message, hiding half in a text prompt and half in the image pixels using GAN for enhanced security.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        hybrid_secret_message = st.text_area(
            "Enter your secret message:",
            "This is a secret message split between prompt and GAN steganography!",
            height=100,
            key="hybrid_message"
        )
        message_length = len(hybrid_secret_message.encode('utf-8')) * 8
        if message_length > 1024:  # Rough estimate for hybrid capacity (512 bits for GAN + prompt capacity)
            st.warning(f"⚠️ Message length ({message_length} bits) may exceed hybrid capacity. Consider a shorter message.")
        
        hybrid_encryption_key = st.text_input(
            "Enter encryption key:",
            "MySecretKey123",
            type="password",
            key="hybrid_key"
        )
        key_valid, key_warning = validate_key(hybrid_encryption_key)
        if key_warning:
            st.warning(key_warning)
        
        cover_image = st.file_uploader(
            "Upload optional cover image (PNG, JPG):",
            type=["png", "jpg", "jpeg"],
            help="Used if prompt-based image generation fails.",
            key="hybrid_cover_image"
        )
        
        if cover_image:
            try:
                cover_img = Image.open(cover_image).convert("RGB")
                if max(cover_img.size) > 2048:
                    st.error("Image is too large. Please upload an image smaller than 2048x2048 pixels.")
                else:
                    if max(cover_img.size) > 512:
                        cover_img = cover_img.resize(
                            (512, int(512 * cover_img.size[1] / cover_img.size[0]))
                            if cover_img.size[0] > cover_img.size[1]
                            else (int(512 * cover_img.size[0] / cover_img.size[1]), 512)
                        )
                    st.image(cover_img, caption="Cover Image", use_column_width=True)
                    st.session_state.hybrid_cover_image = cover_img
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        use_seed = st.checkbox("Use fixed seed for reproducibility", value=True, key="hybrid_seed")
        seed = st.number_input("Seed value:", value=42, disabled=not use_seed, key="hybrid_seed_value")
        
        if st.button("Hide Message", key="generate_hybrid"):
            if not hybrid_secret_message:
                st.warning("Please enter a secret message.")
            elif not hybrid_encryption_key:
                st.warning("Please enter an encryption key.")
            else:
                try:
                    with st.spinner("Hiding message using hybrid prompt-GAN method..."):
                        start_time = time.time()
                        stego_image, prompt = stego_system.encrypt_and_hide_hybrid(
                            hybrid_secret_message,
                            hybrid_encryption_key,
                            cover_image=st.session_state.get('hybrid_cover_image', None),
                            seed=seed if use_seed else None
                        )
                        token_count = len(prompt.split())
                        if token_count > 60:
                            st.warning(f"⚠️ Prompt has {token_count} tokens, may exceed Stable Diffusion's limit (77 tokens).")
                        elapsed_time = time.time() - start_time
                        st.session_state.hybrid_stego_image = stego_image
                        st.session_state.hybrid_prompt = prompt
                        st.success(f"✅ Message hidden in {elapsed_time:.2f} seconds!")
                except Exception as e:
                    st.error(f"Error hiding message: {e}")
    
    with col2:
        st.subheader("Output")
        if 'hybrid_stego_image' in st.session_state:
            st.image(
                st.session_state.hybrid_stego_image,
                caption="Image with hidden message (Prompt + GAN)",
                use_column_width=True
            )
            buf = io.BytesIO()
            st.session_state.hybrid_stego_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Stego Image",
                data=byte_im,
                file_name="stego_image_hybrid.png",
                mime="image/png",
                key="download_hybrid_image"
            )
            with st.expander("View Generated Prompt"):
                st.code(st.session_state.hybrid_prompt)
            st.button(
                "Copy Prompt to Clipboard",
                help="Copy the prompt to share with the recipient",
                key="copy_hybrid_prompt"
            )
    
    st.markdown("---")
    st.subheader("Extract Message")
    hybrid_prompt = st.text_area(
        "Enter the prompt containing the hidden message:",
        height=150,
        key="extract_hybrid_prompt"
    )
    
    hybrid_stego_image = st.file_uploader(
        "Upload stego image with hidden message:",
        type=["png", "jpg", "jpeg"],
        key="extract_hybrid_image"
    )
    
    if hybrid_stego_image:
        try:
            extract_stego_img = Image.open(hybrid_stego_image).convert("RGB")
            if max(extract_stego_img.size) > 2048:
                st.error("Image is too large. Please upload an image smaller than 2048x2048 pixels.")
            else:
                if max(extract_stego_img.size) > 512:
                    extract_stego_img = extract_stego_img.resize(
                        (512, int(512 * extract_stego_img.size[1] / extract_stego_img.size[0]))
                        if extract_stego_img.size[0] > extract_stego_img.size[1]
                        else (int(512 * extract_stego_img.size[0] / extract_stego_img.size[1]), 512)
                    )
                st.image(extract_stego_img, caption="Stego Image", width=300)
                st.session_state.hybrid_extract_stego_image = extract_stego_img
        except Exception as e:
            st.error(f"Error loading image: {e}")
    
    hybrid_extraction_key = st.text_input(
        "Enter the decryption key:",
        type="password",
        key="extract_key_hybrid"
    )
    key_valid, key_warning = validate_key(hybrid_extraction_key)
    if key_warning:
        st.warning(key_warning)
    
    if st.button("Extract Message", key="extract_hybrid_btn"):
        if not hybrid_prompt or not hybrid_stego_image:
            st.warning("Please provide both a prompt and a stego image to extract from.")
        elif not hybrid_extraction_key:
            st.warning("Please enter a decryption key.")
        else:
            try:
                with st.spinner("Extracting hidden message..."):
                    extracted_message = stego_system.extract_and_decrypt_hybrid(
                        hybrid_prompt,
                        st.session_state.hybrid_extract_stego_image,
                        hybrid_extraction_key
                    )
                    if extracted_message:
                        st.success("Message extracted successfully!")
                        st.info(f"**Extracted message:** {extracted_message}")
                    else:
                        st.error("Failed to extract message. Check your prompt, image, and decryption key.")
            except Exception as e:
                st.error(f"Error during extraction: {e}")

# Tab 4: About page
with tab4:
    st.header("About Hybrid Steganography System")
    st.markdown("""
    ## What is Steganography?
    Steganography is the practice of concealing information within other non-secret data to avoid detection. Unlike encryption, which makes data unreadable, steganography hides the existence of the secret message.

    ## This Application
    This system offers three steganography approaches:

    ### 1. Prompt-based Steganography
    - **How it works**: Encodes binary data into text prompts
    - **Generation**: Uses Stable Diffusion to generate images from prompts
    - **Recovery**: Data is recovered from the prompt alone
    - **Advantages**: No image manipulation required

    ### 2. GAN-based Steganography
    - **How it works**: Embeds data in image RGB channels using a neural network
    - **Training**: Uses adversarial training for optimal embedding
    - **Security**: Integrates AES-256 encryption
    - **Advantages**: High capacity and resistance to detection

    ### 3. Hybrid Prompt-GAN Steganography
    - **How it works**: Splits the message, encoding half in a prompt and half in the image
    - **Generation**: Uses Stable Diffusion for the prompt and GAN for the image
    - **Recovery**: Combines prompt and image extraction
    - **Advantages**: Enhanced security and balanced capacity

    ### Technical Features
    - AES-256 encryption for secure message hiding
    - RGB channel optimization for data hiding
    - Semantic encoding for natural-looking prompts
    - Attention mechanisms for improved feature extraction

    ### Use Cases
    - Secure communication
    - Digital watermarking
    - Privacy protection
    - Information authentication
    """)
    st.markdown("---")
    st.caption("Developed as a demonstration of modern steganography techniques.")

# Footer
st.markdown("---")
st.caption("⚠️ Note: This application is for educational purposes only. Please use responsibly and respect privacy laws.")

