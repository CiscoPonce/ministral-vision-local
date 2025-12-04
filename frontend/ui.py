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
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <svg width="80" height="56" viewBox="0 0 365 258" fill="none" xmlns="http://www.w3.org/2000/svg" style="display: inline-block; margin: 0 10px; vertical-align: middle;">
                      <g id="Mistral AI Logo">
                        <path d="M104.107 0H52.0525V51.57H104.107V0Z" fill="#FFD800"/>
                        <path d="M312.351 0H260.296V51.57H312.351V0Z" fill="#FFD800"/>
                        <path d="M156.161 51.5701H52.0525V103.14H156.161V51.5701Z" fill="#FFAF00"/>
                        <path d="M312.353 51.5701H208.244V103.14H312.353V51.5701Z" fill="#FFAF00"/>
                        <path d="M312.356 103.14H52.0525V154.71H312.356V103.14Z" fill="#FF8205"/>
                        <path d="M104.107 154.71H52.0525V206.28H104.107V154.71Z" fill="#FA500F"/>
                        <path d="M208.228 154.711H156.174V206.281H208.228V154.711Z" fill="#FA500F"/>
                        <path d="M312.351 154.711H260.296V206.281H312.351V154.711Z" fill="#FA500F"/>
                        <path d="M156.195 206.312H0V257.882H156.195V206.312Z" fill="#E10500"/>
                        <path d="M364.439 206.312H208.244V257.882H364.439V206.312Z" fill="#E10500"/>
                      </g>
                    </svg>
                    <img src="https://avatars.githubusercontent.com/u/150920049?s=200&v=4" height="80" style="display: inline-block; margin: 0 10px; vertical-align: middle;">
                    <h1 style="color: #4f46e5; margin-top: 10px;">Ministral Vision</h1>
                </div>
            """)
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
