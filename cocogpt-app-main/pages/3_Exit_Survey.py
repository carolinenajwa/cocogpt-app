import streamlit as st
from azure.cosmos import CosmosClient, PartitionKey
from uuid import uuid4

# Cosmos DB credentials
url = st.secrets["COSMO_URL"]
key = st.secrets["COSMO_KEY"]
database_name = st.secrets["COSMO_DB"]
container_name = st.secrets["COSMO_SURVEY_CONTAINER"]

# Initialize the Cosmos client
client = CosmosClient(url, credential=key)
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

def insert_survey_feedback(feedback_data):
    try:
        response = container.upsert_item(feedback_data)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    

# Optional Exit Survey Page
st.title('CocoGPT Exit Survey')

st.markdown("""
    Thank you for participating! 
    We'd love to hear your thoughts.
""")

# Survey Questions
user_id = st.text_input("Your User ID (Optional):")

q1 = st.slider("Rate your overall experience with CocoGPT", 1, 5)
q2 = st.radio("How well did CocoGPT understand the context of your queries?", 
              ['Very Well', 'Somewhat Well', 'Poorly', 'Not at All'])
q3 = st.radio("Was CocoGPT's emotional response appropriate to the conversation?",
              ['Always Appropriate', 'Sometimes Appropriate', 'Rarely Appropriate', 'Never Appropriate'])
q4 = st.radio("How did CocoGPT handle sensitive or critical topics?", 
              ['Handled Well', 'Handled Adequately', 'Handled Poorly', 'Did Not Handle'])
q5 = st.radio("Did CocoGPT avoid providing specific website names, app names, or locations when needed?",
              ['Yes, Always', 'Mostly', 'Sometimes', 'No'])
q6 = st.radio("Did the conversation flow naturally without repetition or abrupt topic changes?",
              ['Yes, Always', 'Mostly', 'Sometimes', 'No'])
q6 = st.radio("Did CocoGPT identify your attempts at ending the conversation and respond appropriately?",
              ['Yes, Always', 'Mostly', 'Sometimes', 'No'])
q7 = st.text_area("Describe any specific issues or errors you encountered:")
q8 = st.text_area("What behavior/response improvements or features would you suggest for CocoGPT?")

# Submit Button
if st.button('Submit Feedback'):
    feedback_data = {
        'id': str(uuid4()),  # Unique ID for each feedback submission
        'UserID': user_id,
        'OverallExperience': q1,
        'ContextUnderstanding': q2,
        'EmotionalResponseAppropriateness': q3,
        'HandlingSensitiveTopics': q4,
        'AvoidingSpecificReferences': q5,
        'ConversationFlow': q6,
        'SpecificIssuesOrErrors': q7,
        'SuggestionsForImprovement': q8
    }
    
    result = insert_survey_feedback(feedback_data)
    if result.get('status') == 'success':
        st.success("Thank you for your feedback!")
    else:
        st.error(f"An error occurred while submitting your feedback: {result.get('error')}")
# Optional Restart or Exit
# restart = st.button("Restart CocoGPT")
# if restart:
#     st.experimental_rerun()

st.markdown("*Your feedback will help us enhance CocoGPT for a better user experience.*")

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: grey;">
        <p style="margin: 5px;">© 2023 CocoGPT - All Rights Reserved</p>
        <p style="margin: 5px;">Developed by 'Responsible Health AI Lab' (RHAIL)</p>
        <p style="margin: 5px;">For support or inquiries, contact us at <a href="mailto:cocobot@uw.edu">cocobot@uw.edu</a></p>
        <p style="margin: 5px;">Learn about us: <a href="https://www.aim-ahead.net/university-of-washington-tacoma/">AIM-AHEAD: Responsible Health AI Lab (RHAIL)</a></p>
    </div>
""", unsafe_allow_html=True)
