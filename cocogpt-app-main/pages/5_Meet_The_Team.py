import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Page Under Construction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# # Load your logo
# logo = Image.open("path_to_your_logo.png")  # Replace with your logo image path

st.title("Page Under Construction")


# Under Construction Message
st.markdown("""
    <div style="text-align: center;">
        <h2>This page is currently under construction.</h2>
        <p>We are working hard to bring you new content soon.</p>
        <p>Stay tuned!</p>
    </div>
    """, unsafe_allow_html=True)

# Optional: Additional Information or Placeholder
# st.info("For more information or updates, visit our [Homepage](URL).")  # Replace URL with your website linkst.markdown("---")
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: grey;">
        <p style="margin: 5px;">© 2023 CocoGPT - All Rights Reserved</p>
        <p style="margin: 5px;">Developed by 'Responsible Health AI Lab' (RHAIL)</p>
        <p style="margin: 5px;">For support or inquiries, contact us at <a href="mailto:cocobot@uw.edu">cocobot@uw.edu</a></p>
    </div>
""", unsafe_allow_html=True)
