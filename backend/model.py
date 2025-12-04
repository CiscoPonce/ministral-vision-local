import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from backend.image_ops import encode_image

class MinistralInference:
    def __init__(self, model_id, tokenizer_id):
        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = MistralCommonBackend.from_pretrained(tokenizer_id)
        
        print(f"Loading model from {model_id}...")
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    def chat(self, message, history, image):
        """
        Handles the chat interaction.
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
        tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
        
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
        output = self.model.generate(
            **tokenized,
            image_sizes=image_sizes,
            max_new_tokens=1024,
        )[0]

        decoded_output = self.tokenizer.decode(output[len(tokenized["input_ids"][0]):])
        return decoded_output
