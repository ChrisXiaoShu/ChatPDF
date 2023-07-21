from prisma import config
from ai_model import BartenderAI, DetailAI, RecommendAI, SummaryPreferenceAI
from lib.conversation import Role, add_msg_to_serialized_history
import chainlit as cl


@cl.langchain_factory(use_async=True)
def factory():
    # Save the conversation history in the user session
    cl.user_session.set("history", "")
    cl.user_session.set("summary", "")
    cl.user_session.set("recommend", "")
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
    
@cl.action_callback("Detail")
async def on_action(action):
    await action.remove()
    
    recommend = cl.user_session.get("recommend")
    detail_ai = DetailAI()
    msg = await detail_ai.run(drinks=recommend)
    
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
    # extract the recommend drink from the response the drink is in between：and，
    drink = msg.content[msg.content.find("：")+1:msg.content.find("，")]
    cl.user_session.set('recommend', drink)
    if cl.config.run.debug:
        msg.actions = [cl.Action(name="Detail", value="example_value2", description="Click me3!")]
    
    await msg.send()

@cl.langchain_postprocess
async def postprocess(output: str):
    # Sending an action button within a chatbot message
    #check config.run debug if debug mode add summary button else only recommend button
    actions = [cl.Action(name="Recommend Drink!", value="example_value1", description="Click me2!")]
    if cl.config.run.debug:
        actions = [
            cl.Action(name="Summary Preference!", value="example_value", description="Click me!"),
            cl.Action(name="Recommend Drink!", value="example_value1", description="Click me2!")
        ]
        
    history = cl.user_session.get("history")
    history = add_msg_to_serialized_history(history, Role.Human, output['input'])
    history = add_msg_to_serialized_history(history, Role.AI, output['response'])
    cl.user_session.set("history", history)
    await cl.Message(content=output['response'], actions=actions).send()