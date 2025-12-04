import gradio as gr

def create_ui(inference_engine):
    """
    Creates and returns the Gradio Blocks UI.
    """
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
            fn=inference_engine.chat,
            inputs=[text_input, gr.State([]), image_input],
            outputs=output_text
        )
    
    # Set theme and css as attributes for Gradio 6.x compatibility
    demo.theme = theme
    demo.css = css
    
    return demo
