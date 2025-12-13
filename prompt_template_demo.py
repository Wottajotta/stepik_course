from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template = (
    "Отвечай как приветственный ассистент: Привет! {name}!\n\n"
    "Рад видеть тебя в нашем сообществе!\n\n"
    "Здесь ты можешь найти информацию по следующим разделам:\n\n"
    "{topic}\n\n"
    "Если будут вопросы, Обращайся!"
)

dict_for_input = {"name": "Владислав!", "topic": "1. Математика\n2. Программирование\n3. Физика"}

model = ChatOpenAI()

prompt = PromptTemplate(template=template, input_variables=["name", "topic"])
greeting_chain = prompt | model | StrOutputParser()

result1 = greeting_chain.invoke(
   dict_for_input
)
print(result1)

prompt2 = PromptTemplate.from_template(
    "Дай краткую информацию по каждому направлению из {topic}"
)

chain2 = (
    RunnablePassthrough.assign(topic=lambda x: greeting_chain.invoke(x))
    | prompt2
    | model
    | StrOutputParser()
)

result2 = chain2.invoke(
    dict_for_input
)
print(result2)
