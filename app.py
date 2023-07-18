from ai_model import BartenderAI, RecommendAI, SummaryPreferenceAI
import chainlit as cl


@cl.langchain_factory(use_async=True)
def factory():
    # Save the conversation history in the user session
    cl.user_session.set("history", "")
    cl.user_session.set("summary", "")
    bartender = BartenderAI()

    return bartender.get_chain()

@cl.action_callback("Summary Preference!")
async def on_action(action):
    await action.remove()
    
    history = cl.user_session.get("history")
    summary_ai = SummaryPreferenceAI()
    msg = await summary_ai.run(conversation_history=history)
    
    cl.user_session.set("summary", msg.content)
    
    await msg.send()

@cl.action_callback("Recommend Drink!")
async def on_action(action):
    await action.remove()
    
    summary = cl.user_session.get("summary")
    if not summary:
        history = cl.user_session.get("history")
        summary_ai = SummaryPreferenceAI()
        msg = await summary_ai.run(history)
        cl.user_session.set("summary", msg.content)
    
    recommend_ai = RecommendAI()
    msg = await recommend_ai.run(preferences=summary)
    
    await msg.send()

@cl.langchain_postprocess
async def postprocess(output: str):
    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="Summary Preference!", value="example_value", description="Click me!"),
        cl.Action(name="Recommend Drink!", value="example_value1", description="Click me1!"),
    ]
    cl.user_session.set("history", output['history'])
    await cl.Message(content=output['response'], actions=actions).send()