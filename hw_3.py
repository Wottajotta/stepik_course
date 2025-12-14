import logging
from dotenv import load_dotenv

import json
from pydantic import BaseModel, Field, ValidationError, conint

from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


def check_valid(output):
    try:
        return 100 <= output.price <= 1000 and 1 <= output.quantity <= 10
    except Exception:
        return False



# Создаем лог-файл + логгер
logging.basicConfig(
    filename="chat_session.log", 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)

# Подключаем модель
load_dotenv()
model = ChatOpenAI(timeout=15)


# Pydantic-модель
class Item(BaseModel):
    name: str = Field(..., description="Имя товара")
    price: conint(ge=100, le=1000) = Field(..., description="Цена товара")
    quantity: conint(ge=1, le=10) = Field(..., description="Количество товара")
    raiting: conint(ge=1, le=5) = Field(..., description="Рейтинг товара")

# Инициализируем инструкцию и формат
output_parser = PydanticOutputParser(pydantic_object=Item)
format_instructions = output_parser.get_format_instructions()

# Задаем промпт
prompt = PromptTemplate(
    template=(
        "Ответь на вопрос в требуемом формате. Цена должна быть от 100 до 1000, количество от 1 до 10.\n"
        "{format_instructions}\n"
        "Вопрос: {user_question}\nОтвет:"
    ),
    input_variables=["user_question"],
    partial_variables={"format_instructions": format_instructions},
)

# Цепочка
chain = prompt | model | output_parser

try:
    question = input("Введите вопрос о товаре: ")
    logging.info(f"User: {question}")
    attempts = 0
    max_attempts = 5

    while attempts < max_attempts:
        attempts += 1
        try:
            result = chain.invoke({"user_question": question})
            logging.info(f"Bot: {result.model_dump_json(ensure_ascii=False)}")
            if check_valid(result):
                print(result.model_dump_json(ensure_ascii=False))
                break
            error_payload = {
                "error": "Неверные значения полей",
                "details": {
                    "price": getattr(result, "price", None),
                    "quantity": getattr(result, "quantity", None),
                },
                "attempt": attempts,
            }
            print(json.dumps(error_payload, ensure_ascii=False))
        except (ValidationError, OutputParserException) as err:
            if isinstance(err, ValidationError):
                details = err.errors()
            else:
                details = str(err)
            print(json.dumps({"error": "Неверный формат ответа", "details": details, "attempt": attempts}, ensure_ascii=False))
    else:
        print(json.dumps({"error": "Не удалось получить корректный ответ", "attempts": max_attempts}, ensure_ascii=False))
except Exception as e:
    print(e)
