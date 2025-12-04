import gradio as gr
import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
import base64
from io import BytesIO
from PIL import Image

# Model Configuration
MODEL_ID = "unsloth/Ministral-3-8B-Reasoning-2512-bnb-4bit"
TOKENIZER_ID = "mistralai/Ministral-3-8B-Reasoning-2512"

print(f"Loading tokenizer from {TOKENIZER_ID}...")
tokenizer = MistralCommonBackend.from_pretrained(TOKENIZER_ID)

print(f"Loading model from {MODEL_ID}...")
model = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

def encode_image(image_path):
    """Encodes an image to a base64 data URI."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def chat(message, history, image):
    """
    Handles the chat interaction.
    message: The user's text message.
    history: Chat history (not used directly here as we construct fresh messages for simplicity, 
             but could be used for multi-turn).
    image: The uploaded image file path (from Gradio).
    """
    
    content = [{"type": "text", "text": message}]
    
    if image:
        image_url = encode_image(image)
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    
    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]

    print("Tokenizing...")
    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
    
    # Move tensors to CUDA and cast floating point to bfloat16
    for k, v in tokenized.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                tokenized[k] = v.to(device="cuda", dtype=torch.bfloat16)
            else:
                tokenized[k] = v.to("cuda")

    # Calculate image_sizes if pixel_values are present
    image_sizes = None
    if "pixel_values" in tokenized:
        image_sizes = [tokenized["pixel_values"].shape[-2:]]

    print("Generating response...")
    output = model.generate(
        **tokenized,
        image_sizes=image_sizes,
        max_new_tokens=1024,
    )[0]

    decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
    return decoded_output

# Gradio Interface
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

css = """
.container { max-width: 900px; margin: auto; padding-top: 2rem; }
h1 { text-align: center; margin-bottom: 1rem; color: #4f46e5; }
.output-markdown { padding: 1rem; background: white; border-radius: 8px; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1); }
"""

with gr.Blocks(title="Ministral Vision") as demo:
    with gr.Column(elem_classes=["container"]):
        gr.Markdown("# 🧠 Ministral Vision")
        gr.Markdown("Upload an image and ask questions to leverage the reasoning capabilities of Ministral-3-8B.")
        
        with gr.Group():
            image_input = gr.Image(type="filepath", label="Input Image", height=450)
            text_input = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., Describe the main object in this image.", 
                lines=2,
                scale=4
            )
            submit_btn = gr.Button("✨ Analyze Image", variant="primary", scale=1)

        gr.Markdown("### 💡 Model Response")
        output_text = gr.Markdown(elem_classes=["output-markdown"])

    submit_btn.click(
        fn=chat,
        inputs=[text_input, gr.State([]), image_input],
        outputs=output_text
    )

demo.theme = theme
demo.css = css

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
