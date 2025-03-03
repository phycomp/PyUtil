對話提示模板 - ChatPromptTemplate
先建立ChatPromptTemplate物件，並代入以串列形式建立而成的對話訊息，
再使用 from_messages 方法建立對話提示模板，要以tuple去建立個別角色的訊息，
並將要替換的部分設為模板參數，最後包起來形成對話訊息串列

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system","你是一位很會教{topic}的老師."),
    ("human","可以再說一次嗎?"),
    ("ai","好的，我再講一次"),
    ("human","{input}"),
])

print(chat_template)
input_variables=['input', 'topic'] messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['topic'], template='你是一位很會教{topic}的老師.')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='可以再說一次嗎?')), AIMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='好的，我再講一次')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]
使用.format_messages()替換對話訊息中的參數
messages_list = chat_template.format_messages(topic="數學",input="甚麼是三角函數?")

print(messages_list)
[SystemMessage(content='你是一位很會教數學的老師.'), HumanMessage(content='可以再說一次嗎?'), AIMessage(content='好的，我再講一次'), HumanMessage(content='甚麼是三角函數?')]
print(chat_model.invoke(messages_list).content)
三角函數是一組描述角度和直角三角形之間關係的數學函數。常見的三角函數包括正弦函數（sine）、餘弦函數（cosine）、正切函數（tangent）、割函數（secant）、余割函數（cosecant）和角度的正弦函數（versine）、餘弦函數（coversine）、正割函數（haversine）等。三角函數在數學和科學中有廣泛的應用，例如在測量、工程、物理等領域。
# 也可使用invoke方法以字典形式代入參數
messages_value = chat_template.invoke({"topic":"數學","input":"甚麼是三角函數?"})

print(messages_value)
print(f"messages_string: {type(messages_list)}, \n"
    f"messages_value: {type(messages_value)}")
messages=[SystemMessage(content='你是一位很會教數學的老師.'), HumanMessage(content='可以再說一次嗎?'), AIMessage(content='好的，我再講一次'), HumanMessage(content='甚麼是三角函數?')]
messages_string: <class 'list'>, 
messages_value: <class 'langchain_core.prompt_values.ChatPromptValue'>
print(chat_model.invoke(messages_value).content)
三角函數是一組描述角度和直角三角形之間關係的數學函數。常見的三角函數包括正弦函數（sine）、餘弦函數（cosine）、正切函數（tangent）、割函數（secant）、余割函數（cosecant）和角度的正弦函數（versine）、餘弦函數（coversine）、正割函數（haversine）等。三角函數在數學和科學中有廣泛的應用，例如在測量、工程、物理等領域。
角色物件
from langchain.schema import AIMessage,HumanMessage,SystemMessage

prompt = ChatPromptTemplate(
        messages = [SystemMessage(content = "你是一名籃球員"),
                    HumanMessage(content = "我上籃都放槍"),
                    AIMessage(content = "你為甚麼都放槍?")]
)

print(prompt)
input_variables=[] messages=[SystemMessage(content='你是一名籃球員'), HumanMessage(content='我上籃都放槍'), AIMessage(content='你為甚麼都放槍?')]
prompt = (SystemMessage(content = "你是一名籃球員") +
                    HumanMessage(content = "我上籃都放槍") +
                    AIMessage(content = "你為甚麼都放槍?"))

print(prompt)
因為這邊沒有模板參數，所以input_variables為空值
    input_variables=[] messages=[SystemMessage(content='你是一名籃球員'), HumanMessage(content='我上籃都放槍'), AIMessage(content='你為甚麼都放槍?')]
一樣使用format_message()方法，而因為沒有模板參數，所以不需要傳入任何模板參數，
就可直接得到messages串列，最後傳給模型得到回覆
print(chat_model.invoke(prompt.format_messages()).content)
或許你可以試著在訓練中專注於提升你的射籃技巧，包括姿勢、手感和力量控制。透過持之以恆的練習和專注，你可以提高你的籃球射門命中率。另外，也可以找教練或隊友給予建議和指導，幫助你改善射籃技巧。記得，籃球是一項需要不斷努力和改進的運動，相信自己，堅持下去，你一定能夠取得進步的！
訊息提示模板
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
使用 + 也可以串接message prompt template
prompt = (SystemMessage(content = "你是一名籃球員") +
        HumanMessage(content = "我上籃都放槍") +
        AIMessage(content = "你為甚麼都放槍?") +
        HumanMessagePromptTemplate.from_template("{input}")
        )

print(prompt)
input_variables=['input'] messages=[SystemMessage(content='你是一名籃球員'), HumanMessage(content='我上籃都放槍'), AIMessage(content='你為甚麼都放槍?'), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]
print(prompt.format_messages(input = "我不會左手上籃"))
[SystemMessage(content='你是一名籃球員'), HumanMessage(content='我上籃都放槍'), AIMessage(content='你為甚麼都放槍?'), HumanMessage(content='我不會左手上籃')]
也可省略HumanMessagePromptTemplate()這一長串，直接放入參數(用{}包好即可)
prompt = (SystemMessage(content = "你是一名籃球員") +
        HumanMessage(content = "我上籃都放槍") +
        AIMessage(content = "你為甚麼都放槍?") +
        "{input}"
        )

print(prompt)
input_variables=['input'] messages=[SystemMessage(content='你是一名籃球員'), HumanMessage(content='我上籃都放槍'), AIMessage(content='你為甚麼都放槍?'), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}'))]
不用代參數，就可以在提示語中回答
from langchain.prompts import PromptTemplate


prompt = PromptTemplate(template="試著以{role}的角度," "告訴我一個關於{topic}的知識",input_variables=["role", "topic"])
print(prompt)
input_variables=['role', 'topic'] template='試著以{role}的角度,告訴我一個關於{topic}的知識'
法一: 使用partial代入固定模板
partial_prompt = prompt.partial(topic="大便")
print(partial_prompt)
input_variables=['role'] partial_variables={'topic': '大便'} template='試著以{role}的角度,告訴我一個關於{topic}的知識'
此時剩下role此參數還未被指定而已，只需要再帶入role參數為何即可
print(partial_prompt.format(role="醫生對民眾"))
試著以醫生對民眾的角度,告訴我一個關於大便的知識
# 法二: 直接在模板中使用partial_variables加入固定參數
prompt = PromptTemplate(template="試著以{role}的角度," "告訴我一個關於{topic}的知識",input_variables=["role"],partial_variables={"topic":"洗手"})
print(prompt)
input_variables=['role'] partial_variables={'topic': '洗手'} template='試著以{role}的角度,告訴我一個關於{topic}的知識'
print(prompt.format(role="醫生對民眾"))
試著以醫生對民眾的角度,告訴我一個關於洗手的知識
在對話提示模板，也可以使用partial來固定參數
from langchain_core.prompts import ChatPromptTemplate
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","試著以{role}的角度說明"),
        ("human","告訴我一個關於{topic}的知識"),
    ]
)

print(chat_template)

chat_partial_prompt = chat_template.partial(topic="大便")
print(chat_partial_prompt)

print(chat_partial_prompt.format_messages(role="醫生對民眾"))
input_variables=['role', 'topic'] messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['role'], template='試著以{role}的角度說明')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['topic'], template='告訴我一個關於{topic}的知識'))]
input_variables=['role'] partial_variables={'topic': '大便'} messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=['role'], template='試著以{role}的角度說明')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['topic'], template='告訴我一個關於{topic}的知識'))]
[SystemMessage(content='試著以醫生對民眾的角度說明'), HumanMessage(content='告訴我一個關於大便的知識')]
以函式自動加入最新內容
import pytz
from datetime import datetime
timezone = pytz.timezone("Asia/Taipei")
# 先定義好當前時間函數
def get_datetime():
    now = datetime.now(timezone)
    return now.strftime("%Y/%m/%d,%H:%M:%S")

print(get_datetime())
2024/08/09,01:06:43
prompt = PromptTemplate.from_template("現在時間是:{date}")
partial_prompt = prompt.partial(date=get_datetime())
print(partial_prompt)

print(partial_prompt.format())
input_variables=[] partial_variables={'date': '2024/08/09,01:06:43'} template='現在時間是:{date}'
現在時間是:2024/08/09,01:06:43
partial_prompt = prompt.partial(date=get_datetime())
print(partial_prompt.format())
現在時間是:2024/08/09,01:06:43
提示模板中的佔位訊息
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.prompts import MessagesPlaceholder

human_prompt = "用{word_count}個字來總結我們迄今為止的對話"
human_message_template =HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="conversation"),human_message_template])

print(chat_prompt)
input_variables=['conversation', 'word_count'] input_types={'conversation': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]]} messages=[MessagesPlaceholder(variable_name='conversation'), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['word_count'], template='用{word_count}個字來總結我們迄今為止的對話'))]
from langchain.schema import AIMessage,HumanMessage,SystemMessage # 角色物件

human_message = HumanMessage(content="學習程式設計的最佳方法是甚麼?")
ai_message = AIMessage(content = """
1. 選擇程式語言: 決定想學習的程式語言
2. 想解決的問題: 遇到那些想要解決的問題
3. 打好基礎: 熟悉語法、資料結構等
"""
)

new_chat_prompt = chat_prompt.format_prompt(conversation=[human_message,ai_message],word_count="20")
print(new_chat_prompt)

print(chat_model.invoke(new_chat_prompt).content)
messages=[HumanMessage(content='學習程式設計的最佳方法是甚麼?'), AIMessage(content='\n1. 選擇程式語言: 決定想學習的程式語言\n2. 想解決的問題: 遇到那些想要解決的問題\n3. 打好基礎: 熟悉語法、資料結構等\n'), HumanMessage(content='用20個字來總結我們迄今為止的對話')]
學習程式設計需選擇語言、解決問題、打好基礎，持續實踐與學習。
那這樣輸出格式有辦法自訂嗎?
當然有ㄚ
下一次要來講Output Parsers囉!
