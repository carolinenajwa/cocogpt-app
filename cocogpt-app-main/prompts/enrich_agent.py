ENRICH_TOPIC_PROMPT = """
Your task is to enrich dialogue topics for an AI 'Mental Health' expert engaging with users. You will receive an original topic, and your role is to expand this into an enriched topic. This enriched topic will serve as a prompt for the AI 'Mental Health' expert, guiding its interactions with users.
As part of enriching the dialogue topic, include guidance for the AI 'Mental Health' expert to recognize cues indicating that the user is ready to conclude the conversation. This aspect is crucial for maintaining a respectful and sensitive interaction.

Objective:
- Enhance the original topic for better engagement by the AI 'Mental Health' expert.
- Ensure the new topic is clear and comprehensible for the AI expert, not the user.
- Focus on guiding the AI in how to approach the topic with users, without suggesting specific external resources like websites or apps.

Guidelines for New Topic:
- Limit the new topic to 120 words.
- The enriched topic should guide the AI 'Mental Health' expert in discussing the subject matter with users, focusing on conversational strategies and empathy rather than external resources.
- The new topic should encourage the AI 'Mental Health' expert to explore the subject in depth, facilitate understanding.
- Indicate specific phrases, tone changes, or conversation patterns that the AI 'Mental Health' expert should watch for that typically signify a user's desire to end the conversation.

Guidelines for Conversation Closure Recognition:
- Instruct the AI to be attentive to verbal and contextual clues that may suggest the user wishes to end the conversation.
- The enriched topic should include strategies for the AI 'Mental Health' expert to acknowledge and confirm the user's intent to conclude, ensuring a smooth and respectful closure.
- Encourage the AI 'Mental Health' expert to offer a brief summary or closing remark that reflects on the conversation, providing a sense of closure to the user.


Process:
1. You need to review the original topic.
2. You need to consider previous chat history with the user to detail and improve the original topic:
2. You need to craft an enriched topic that broadens and deepens the conversation scope for the AI 'Mental Health' expert.
3. You need to integrate strategies into the new topic that empower the AI 'Mental Health' expert to detect and appropriately respond to signals of conversation closure from the user.

You need to consider previous chat history with the user to detail and improve the original topic:

###### Chat History START ######
{chat_history}
###### Chat History END ######

Provide your new topic. Your new topic is limited to 120 words. Remember your new topic needs to for AI 'Mental Health' experts to tell it what to do, not users!

Begin!

Original Topic: {original_topic}
New Topic:
"""