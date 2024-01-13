from queue import Queue, Full  # Import Full here
from langchain.callbacks.base import BaseCallbackHandler

class StreamingQueueCallbackHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            self.q.put(None, timeout=1)  # Signal end of stream
        except Full:
            print("Warning: Queue is full. Unable to signal end of stream.")
