# welcome.py
import streamlit as st
from azure.cosmos import CosmosClient, PartitionKey

# Initialize Cosmos DB (use the same credentials as in app.py)
url = st.secrets["COSMO_URL"]
key = st.secrets["COSMO_KEY"]
database_name = st.secrets["COSMO_DB"]
container_name = st.secrets["COSMO_CONTAINER"]

client = CosmosClient(url, credential=key)
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

# Function to insert user information into Cosmos DB
def insert_user_info(user_data):
    try:
        response = container.upsert_item(user_data)
        return response
    except Exception as e:
        return {"error": str(e)}


def check_auth_token():
    entered_code = st.session_state.get("auth_code", "")
    if entered_code == st.secrets["AUTH_TOKEN"]:
        st.session_state['auth_status'] = True
        st.session_state['show_auth'] = False  # Hide authentication panel
        auth_message_placeholder.success("Authentication successful!")
        st.rerun()
    else:
        st.session_state['auth_status'] = False
        auth_message_placeholder.error("Incorrect authentication code.")


# Function to display instructions
def display_instructions():
    st.header("Welcome to CocoGPT Testing UI!")

    st.markdown("""
      **Estimated Testing Time: 15 Minutes**  
        Please note that testing should take no more than 15 minutes, unless you wish to explore further. Aim for about 5-10 turns during your conversation with CocoGPT.

        We're thrilled to have you on board to test CocoGPT, our conversational AI designed for family caregivers. 
        Let's get you up to speed with some helpful guidelines:

        ## Getting Started 💬
        1. **Thoroughly Review the Problem-Solving Therapy (PST):** This is a crucial aspect of using CocoGPT effectively. You'll find a general overview of the PST structure on the Welcome Page and in depth explaination in the 'Problem Solving Therapy (PST)' section in the left side bae
        2. **Carefully Read Key Points to Note:** These are essential insights. You'll find them right here on the Welcome page.
        2. **Accessing the Chat:** Look for the "Chat with CocoGPT" section in the left sidebar.
        3. **Completing the Exit Survey:** Don’t forget to share your thoughts in the "Exit Survey" section, also in the left sidebar.

        ## Problem-Solving Therapy Structure 🧠
        CocoGPT is designed to follow the core principals of problem-solving therapy (PST) structure. This approach empowers users by cultivating self-reliance problem solving. The structure encompasses:
        1. Selecting and defining the problem.
        2. Establishing realistic and achievable goals for problem resolution.
        3. Generating alternative solutions.
        4. Implementing decision-making guidelines.
        5. Evaluation and choosing solutions.
        6. Implementing the preferred solutions.
        7. Evaluating the outcome.
        Please note, CocoGPT **does not** currently include step 7 (evaluating the outcome) of the PST structure. 

        ## Key Points to Note  🔍
        **✅ Provides Empathetic Responses:** Is CocoGPT responding in a friendly, supportive, and empathetic tone? \n
        **✅ Handles Sensitive Topic Appropriately:** How does CocoGPT navigate through sensitive mental health topics? \n
        **✅ Acknowledges Conversation Wrap-up:** How well does CocoGPT conclude conversations? Does it feel natural and complete? \n
        **✅ Maintains Natural Conversation Flow:** Is the chat progressing smoothly and naturally? Watch out for any odd repetitions or abrupt topic shifts. \n
        **❌ Avoids Medical or Legal Advice Disclaimer:** Does CocoGPT refrain from offering professional medical or legal advice. \n
        **❌ Avoids Real-world Reference Avoidance:** Does CocoGPT steer clear of suggesting specific websites, apps, or locations? \n
                
        ## Your Feedback is Precious 🌟
        - **Rating the Experience:** Please use the feedback buttons following CocoGPT's responses to share how you felt about the interaction.
        - **Sharing Additional Thoughts:** We welcome your detailed feedback in the "Optional Additional Feedback" section.
        - **Survey about your Experience:** Please complete the survey upon completion locatedin the "Exit Survey" section.

        Your input is vital in enhancing CocoGPT for everyone. We appreciate your time and insights. Enjoy your testing experience! 
    """, unsafe_allow_html=True)

def load_css(filename):
    with open(filename, "r") as f:
        return f"<style>{f.read()}</style>"

# Set page configuration and load CSS
st.set_page_config(page_title="Welcome to CocoGPT's Testing Interface", layout="wide")
css_file = "styles.css"
st.markdown(load_css(css_file), unsafe_allow_html=True)

# Initialize session state variables
if 'auth_status' not in st.session_state:
    st.session_state['auth_status'] = False
if 'info_submitted' not in st.session_state:
    st.session_state['info_submitted'] = False
if 'show_auth' not in st.session_state:
    st.session_state['show_auth'] = True  # Initially, show the authentication panel

# Authentication Panel
if not st.session_state['info_submitted'] and st.session_state['show_auth']:
    col1, col2, col3 = st.columns([1.5,1.5,1])
    with col2:
        st.image('images/welcome_icon.png', width=200)

    col4, col5, col6 = st.columns([1, 2, 1])
    with col5:
        st.header("Authentication")
        auth_message_placeholder = st.empty()
        entered_auth_code = st.text_input("Enter Authentication Token", type="password", key="auth_code")
        if st.button("Check Authentication Token", key="auth_button"):
            check_auth_token()

# User Information Section (display only if authenticated and info not yet submitted)
if st.session_state.get('auth_status', False) and not st.session_state['info_submitted']:
    st.header("User Information")
    temp_user_id = st.text_input("**User ID (required)**", key="temp_user_id")
    first_name = st.text_input("First Name (optional)", key="first_name")
    last_name = st.text_input("Last Name (optional)", key="last_name")
    email = st.text_input("Email (optional)", key="email")

    if st.button("Submit Information"):
        if temp_user_id:
            st.session_state['user_id'] = temp_user_id
            user_data = {
                "id": temp_user_id,
                "UserID": temp_user_id,
                "FirstName": first_name,
                "LastName": last_name,
                "Email": email
            }
            result = insert_user_info(user_data)
            if "error" in result:
                st.error(f"Failed to submit user information: {result['error']}")
            else:
                st.session_state['info_submitted'] = True
                st.success("User information submitted successfully!")
                st.rerun()  # Rerun the app to update the display

# Display instructions if the information has been submitted
if st.session_state['info_submitted']:
    display_instructions()

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: grey;">
        <p style="margin: 5px;">© 2023 CocoGPT - All Rights Reserved</p>
        <p style="margin: 5px;">Developed by 'Responsible Health AI Lab' (RHAIL)</p>
        <p style="margin: 5px;">For support or inquiries, contact us at <a href="mailto:cocobot@uw.edu">cocobot@uw.edu</a></p>
    </div>
""", unsafe_allow_html=True)
