# CocoGPT #

Developed as part of the research conducted by the Machine-Learning for Health Equity lab and the Privacy Preserving Machine-Learning research group at the University of Washington. 

-- last revised 01/2023

------------------------------------------------------------

Overview:
---------
CocoGPT is a conversational AI application designed to provide empathetic and interactive support to family caregivers. Built upon the success of its predecessors, this version integrates enhanced features with Streamlit's dynamic front-end interface and efficient backend data management systems.

Key Features:
--------------
- Enhanced Streamlit Web Interface offering intuitive user interactions.
- Advanced data management utilizing Azure CosmoDB NoSQL database for secure and efficient data handling.
- An improved feedback mechanism enabling users to provide real-time feedback on chatbot interactions.
- Sophisticated empathetic communication strategies implemented for deeper user engagement.
- Upgraded security measures including robust user authentication.

Installation:
-------------
1. Ensure Python is installed on your system and chosen IDE.
2. Install required requirements.txt
3. Configure Streamlit secrets to securely store database credentials and Azure OpenAI API keys.

Usage:
------
- Launch the application using Streamlit: streamlit run Welcome.py
- Access CocoGPT 04 through the provided local URL in Streamlit.
- Utilize secure token-based authentication for user access.

Security and Authentication:
----------------------------
- Implement advanced token-based authentication for user security.
- Store and manage API keys and database credentials using Streamlit secrets for enhanced security.

Troubleshooting:
----------------
- Ensure all dependencies are correctly installed and up to date.
- Confirm the accuracy of database credentials and API key details in Streamlit secrets.
- Regularly check the Streamlit console for potential errors or issues during runtime.

Contributor:
----------------
- Caroline El Jazmi

Inspired By: 
----------------
- Paper Title: [DiagGPT: An LLM-based Chatbot with Automatic Topic Management for Task-Oriented Dialogue](https://arxiv.org/abs/2308.08043)

Questions/Feedback: 
----------------
If you have any questions, feedback, or would like to discuss any aspect of the project, please feel free to [contact me](mailto:eljazmi.contact@gmail.com).
