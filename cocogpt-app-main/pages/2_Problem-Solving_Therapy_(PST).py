import streamlit as st

# Page configuration
st.set_page_config(page_title="Problem-Solving Therapy Overview", layout="wide")

def main():
    st.title("Problem-Solving Therapy (PST) Overview")

    # Introduction to PST
    st.header("What is Problem-Solving Therapy?")
    st.write("""
    Problem-solving therapy refers to a psychological treatment that helps to teach you to effectively manage the negative effects of stressful events that can occur in life. Such stressors can be rather large, such as getting a divorce, experiencing the death of a loved one, losing a job, or having a chronic medical illness like cancer or heart disease. Negative stress can also result from the accumulation of multiple “minor” occurrences, such as ongoing family problems, financial difficulties, constantly dealing with traffic jams, or tense relationships with co-workers or a boss.
    """)

    # Benefits of PST
    st.header("Problem-solving therapy has been found to be effective for a wide range of problems, including:")
    st.write("""
    - Major depressive disorder
    - Generalized anxiety disorder
    - Emotional distress
    - Suicidal ideation
    - Relationship difficulties
    - Certain personality disorders
    - Poor quality of life and emotional distress related to medical illness, such as cancer or diabetes
    """)

    # Skills Developed in PST
    st.header("Skills Developed in PST")
    st.write("""
    - Making effective decisions.
    - Generating creative means of dealing with problems.
    - Accurately identifying barriers to reaching one’s goals.
    """)

    # Goals of PST
    st.header("Goals of Problem-Solving Therapy")
    st.write("""
    - To identify which types of stressors tend to trigger emotions, such as sadness, tension, and anger.
    - Better understand and manage negative emotions.
    - Become more hopeful about your abilities to deal with difficult problems in life.
    - Be more accepting of problems that are unsolvable.
    - Be more planful and systematic in the way you attempt to resolve stressful problems.
    - Be less avoidant when problems occur.
    - Be less impulsive about wanting a “quick fix” solution.
    """)

    # Citation
    st.header("Citation")
    st.write("""
    This information is sourced from the American Psychological Association | Division 12. For more details, visit [their website](https://www.div12.org/sites/default/files/WhatIsProblemSolvingTherapy.pdf).
    """)

if __name__ == "__main__":
    main()
