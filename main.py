from backend.model import MinistralInference
from frontend.ui import create_ui
import config

if __name__ == "__main__":
    # Initialize the inference engine
    engine = MinistralInference(
        model_id=config.MODEL_ID,
        tokenizer_id=config.TOKENIZER_ID
    )

    # Create the UI
    demo = create_ui(engine)

    # Launch the app
    demo.launch(
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT
    )
