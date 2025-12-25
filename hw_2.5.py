import yaml
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


model = ChatOpenAI(temperature=0)


example_prompt = PromptTemplate.from_template(
    "Язык программирования: {programming_language}\nТеоретический вопрос: {theoretical_question}\nПояснение на основе теории: {theoretical_knowledge}\nПример использования в коде:\n{code_example}"
)

with open("examples.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

examples = data["examples"]

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="На основе приведенных примеров, предоставь теоретическое пояснение, затем пример кода для следующего языка программирования.",
    suffix="Язык программирования: {programming_language}\nТеоретический вопрос: {theoretical_question}\nПояснение на основе теории: \nПример использования в коде:",
    input_variables=["programming_language", "theoretical_question"],
)

formatted_prompt = prompt.format(
    programming_language="Python",
    theoretical_question="Как отсортировать список объектов по полю в Python?",
)
response = model.invoke(formatted_prompt)

print(response.content)
