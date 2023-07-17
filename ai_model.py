# create basic AI model class with template model variable and has get_chain and run methods
from itertools import chain
import re
from typing import Any
from click import prompt
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


class AIModel:
    """Base class for AI models."""

    template: str
    prompt: PromptTemplate
    model: BaseLanguageModel

    def get_chain(self, **kwargs: Any) -> Any:
        raise NotImplementedError("get_chain not implemented")

    def run(self, **kwargs: Any) -> Any:
        """Run the model."""
        chain = self.get_chain(**kwargs)
        return chain.run(**kwargs)
    
class BartenderAI(AIModel):
    model = ChatOpenAI(temperature=0)

    template = """The following is a friendly conversation between a human and an AI. The AI is a professional bartender and help human find a cocktail that suits. AI should guide human in choosing a cocktail that is tailored to its preferences. AI should understand human preferences based on human preferred texture, type of alcohol, taste, or personal characteristics. please don't recommend a particular cocktail to human. AI job is merely understand human preference. And don't ask too complex question make question simple and one at a time. 請用繁體中文與我對答案。 
Current conversation:
{history}
Human: {input}
AI:
"""

    prompt = PromptTemplate(template=template, input_variables=["history", "input"])
    
    def get_chain(self, **kwargs: Any) -> Any:
        return ConversationChain(
            prompt=self.prompt,
            llm=self.model,
            memory=ConversationBufferMemory()
        )
            
class SummaryPreferenceAI(AIModel):
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """please summary the human preference from the following conversation in 繁體中文
Current conversation:
{history}
"""
    
    prompt = PromptTemplate(template=template, input_variables=["history"])
    
    def get_chain(self, **kwargs: Any) -> Any:
        return LLMChain(llm=self.model, prompt=self.prompt)
    
    def run(self, conversation_history):
        chain = self.get_chain()
        result = chain.run({ 'history': conversation_history})
        prompt = self.prompt.format(history=conversation_history)
        return result, prompt


class RecommandAI(AIModel):
    model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """
    please choice one of the cocktail from the menu below base on the human preference and reply in 繁體中文
    here is human preference:
    {preferences}
    
    here is the menu:
    1. 青雲閤＄400
    基底：gin琴酒
    Nordes gin諾帝斯琴酒/St.germain接骨木花利口酒/Skinos乳香利口酒/
    Jasmine Syrup自製茉莉花糖漿/Citrus Acid檸檬酸液/tonic通寧水
    酒感2
    口感：微甜/花香味強烈/清爽/氣泡感

    2. 和泉樓＄400
    基底：Gin琴酒
    Generous Gin Azur大方琴酒/Crème de Violet 紫羅蘭利口酒/Lime Juice萊姆汁 /Lavender Syrup自製薰衣草糖漿/ La Caravedo Pisco秘魯白蘭地
    酒感3.5
    口感：偏酸爽口/如同香水的強烈花香

    3. 醉花園＄450
    基底：Homemade Rose Liqueur自製玫瑰利口酒
    Homemade Rose Liqueur自製玫瑰利口酒/Red Repper Syrup粉紅胡椒糖漿/Hendricks Flora Adora Gin亨利爵士花神/Latic Acid乳酸/Cream鮮奶油/
    Egg White蛋白/Soda Water蘇打水
    酒感3
    口感： 蛋糕般的綿密奶泡/主體玫瑰花香帶一絲粉紅胡椒的偏甜辛香

    4. 鐵觀音＄400
    基底：vodka伏特加
    Tieguanyin tea infused vodka鐵觀音伏特加/Cointreau君度橙酒/
    Crème de peach水蜜桃利口酒
    酒感2
    口感：水蜜桃甜香為前調/中調展現鐵觀音培茶風味/清爽的氣泡/酒體輕盈

    5. 文山包種＄400
    基底：Gin琴酒
    Wen Shan Pochong包種茶琴酒/Pavan葡萄利口酒/Lavender Leaf Syrup自製薰衣草片糖漿/Lemon juice檸檬汁
    酒感3
    口感：偏甜爽口/花草香/麝香葡萄與橙花氣味為中調/茶香做為後韻

    6. 金萱＄430
    基底：White Wine白葡萄酒
    Jin Xuan Tea Infused White Wine金萱茶白葡萄酒/Pineapple Sage Infused Apple Whiskey鳳梨鼠尾草蘋果威士忌/Chamomile Cordial洋甘菊風味液/Cream cheese Foam奶油起司泡沫
    酒感3
    口感：上層奶泡起司蛋糕風味呼應金萱茶獨特奶香/中調強烈洋甘菊轉為鼠尾草與蘋果的清新/微苦茶感與葡萄弱酸做結尾
    
    """
    
    prompt = PromptTemplate(template=template, input_variables=["preferences"])
    
    def get_chain(self, **kwargs: Any) -> Any:
        return LLMChain(llm=self.model, prompt=self.prompt)
    
    def run(self, preferences):
        chain = self.get_chain()
        return chain.run({ 'preferences': preferences}), self.prompt.format(preferences=preferences)