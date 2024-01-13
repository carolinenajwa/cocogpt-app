# app.py
import streamlit as st
import uuid
from trubrics.integrations.streamlit import FeedbackCollector
from CocoGPT import CocoGPT  
from streaming_queue_callback import StreamingQueueCallbackHandler  
from azure.cosmos import CosmosClient, PartitionKey
import time
from PIL import Image

# Cosmos DB credentials
url = st.secrets["COSMO_URL"]
key = st.secrets["COSMO_KEY"]
database_name = st.secrets["COSMO_DB"]
container_name = st.secrets["COSMO_CONTAINER"]

# Initialize the Cosmos client
client = CosmosClient(url, credential=key)
# Select the database and container
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

# Initialize CocoGPT instance
my_cocogpt = CocoGPT()

# Initialize styles.css
def load_css(filename):
    with open(filename, "r") as f:
        return f"<style>{f.read()}</style>"

# Initialize script.js     
def load_js(filename):
    with open(filename, "r") as file:
        return f"<script>{file.read()}</script>"

# Set page configuration
st.set_page_config(
    page_title="CocoGPT Training UI",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_file = "styles.css"
js_file = "script.js"
st.markdown(load_css(css_file), unsafe_allow_html=True)
st.markdown(load_js(js_file), unsafe_allow_html=True)

# Initialize feedback collector
collector = FeedbackCollector(email=st.secrets["TRUBRICS_EMAIL"], password=st.secrets["TRUBRICS_PASSWORD"], project=st.secrets["TRUBRICS_PROJECT"])

# Initialize the CocoGPT instance in session_state if it does not exist
if 'cocogpt' not in st.session_state:
    st.session_state['cocogpt'] = CocoGPT()

# Use st.session_state.cocogpt for operations
my_cocogpt = st.session_state['cocogpt']

# Check if the user is authenticated and user ID is entered
auth_status = st.session_state.get('auth_status', False)
user_id_entered = st.session_state.get('user_id', False)

# Create three columns with adjusted widths
col1, spacer, col2 = st.columns([2, 0.5, 0.6])  # Adjust these values as needed
with col1:
    # Chat interface
    st.title("CocoGPT Chat")
with col2:
    st.image("images/no_background_coco.png", width=100)  # Adjust width as needed

# Light gray section for instructions
with st.expander("**Instructions & Disclaimer**", expanded=True):
    st.markdown("""
        <div style="background-color: lightgray; padding: 10px;">
            <p><strong>When interacting with CocoGPT, please assume the role of a family caregiver, engaging in conversations and asking questions as you would in that capacity.</strong></p> 
                <ul>
                    <li>Type your message in the input box below and hit enter or click 'Send'.</li>
                    <li>CocoGPT will respond to your query in real-time.</li>
                    <li>You can provide feedback on responses using the thumbs up or down buttons.</li>
                    <li>If you wish to restart the conversation or encounter any issues hindering the chat flow, click the 'Reset Conversation' button.</li>
                </ul>
            <p><strong>Disclaimer:</strong> CocoGPT can make mistakes. Consider checking important information.</p>
        </div>
        """, unsafe_allow_html=True)
# # Disclaimer
# st.markdown("""
#     <div style="color: grey; font-size: small;">
#         <p><strong>Disclaimer:</strong> CocoGPT is not a replacement for expert medical and mental help. Please consult with a qualified professional for any serious matters.</p>
#     </div>
#     """, unsafe_allow_html=True)

if auth_status and user_id_entered:
    # stream = st.toggle("Stream LLM response", value=True)
    # Check for the existence of 'messages', 'feedback_key', and 'logged_prompt' in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "prompt_ids" not in st.session_state:
        st.session_state["prompt_ids"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    if "feedback_key" not in st.session_state:
        st.session_state["feedback_key"] = 0

    model = "COCO"
    # tags = [f"llm_chatbot{'_stream' if stream else ''}.py"]
    messages = st.session_state.messages

    # Refresh button to increment feedback key and reset logged_prompt
    if st.button("Reset Conversation"):
        my_cocogpt.reset_conversation()  # Reset CocoGPT conversation
        st.session_state['messages'] = []  # Clear conversation history
        st.session_state.feedback_key += 1
        st.session_state.logged_prompt = None  # Reset the logged prompt
        # Use st.rerun() carefully to avoid unintended session resets
        st.rerun()  # Rerun the app to hide the authentication and user info sections


    # Render chat messages
    for n, msg in enumerate(messages):
        st.chat_message(msg["role"]).write(msg["content"])

        if msg["role"] == "assistant" and n:
            feedback_key = f"feedback_key{int(n / 2)}"
            if feedback_key not in st.session_state:
                st.session_state[feedback_key] = None

            # Fetch the corresponding user's prompt and assistant's response
            user_prompt = messages[n-1]["content"] if n-1 >= 0 else ""
            bot_completion = msg["content"]

            feedback = collector.st_feedback(
                component="default",
                feedback_type="thumbs",
                open_feedback_label="Provide additional feedback",
                model=model,
                # tags=tags,
                key=feedback_key,
                prompt_id=st.session_state.prompt_ids[int(n / 2) - 1],
                user_id=st.session_state['user_id'],
                metadata={"user_prompt": user_prompt, "assistant_response": bot_completion}
            )
            # if feedback:
            #     with st.sidebar:
            #         st.write(":orange[Here's the raw feedback you sent to [Trubrics](https://trubrics.streamlit.app/):]")
            #         st.write(feedback)

    # Handle user input
    if prompt := st.chat_input("Type your message here..."):
        messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Process the user input and generate a response
        with st.chat_message("assistant"):
            container = st.empty()
            response = my_cocogpt.run(prompt)  # Use my_cocogpt for processing
            output = response.replace("COCO:", "").replace("AI:", "").replace("COCO (Mental Health Expert):", "")
            messages.append({"role": "assistant", "content": output})
            container.markdown(output)

            logged_prompt = collector.log_prompt(
                config_model={"model": model},
                prompt=prompt,
                generation=output,
                session_id=st.session_state.session_id,
                # tags=tags,
                user_id=st.session_state['user_id']
            )
            st.session_state.prompt_ids.append(logged_prompt.id)
else:
    if not auth_status:
        st.info("Please authenticate to use the CocoGPT.")
    elif not user_id_entered:
        st.info("Please enter a User ID to use the CocoGPT.")
