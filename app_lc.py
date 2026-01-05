import os
import openai
import logging
from dotenv import load_dotenv
from src.brand_chain import BRAND, STYLE
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

logging.basicConfig(
    filename="chat_session.log", 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)


# Создаём класс для CLI-бота
class CliBot():
    # Создаём модель 
    def __init__ (self, model_name):
        self.chat_model = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            timeout=15,
        )
        
        # Хранилище истории по сессиям
        self.store = {}
        
        # Создаём шаблон промта
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Ты — ассистент, который пишет в стиле бренда {BRAND}. В ответе использует не более {STYLE["tone"]['sentences_max']} предложений."
                   f"Тон: {STYLE['tone']['persona']}. Избегай: {', '.join(STYLE['tone']['avoid'])}. "
                   f"Обязательно: {', '.join(STYLE['tone']['must_include'])}."
                   f"В случае отсутствия данных по вопросу используй: {STYLE["fallback"]["no_data"]}."
                   f"В случае неопределённости используй: {STYLE['fallback']['unclear_query']}."
                   f"В случае ошибки используй: {STYLE['fallback']['error']}."),
            MessagesPlaceholder(variable_name='history'),
            ('human', '{question}'),
        ])
        # Создаём цепочку без истории
        self.chain =  self.system_prompt | self.chat_model

        # Создаём цепочку с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain, # Цепочка с историей
            self.get_session_history, # метод для получения истории
            input_messages_key='question', # ключ для вопроса
            history_messages_key='history', # ключ для истории
        )

    # Функция для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def __call__(self, session_id):
        while True:
            try:
                user_text = input("Вы: ").strip()
                logging.info(f"User: {user_text}")
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            msg = user_text.lower()
            if msg in {'выход', 'стоп', 'конец'}:
                print('Бот: До свидания!')
                break
            if msg == 'сброс':
                if session_id in self.store:
                    del self.store[session_id]
                print('Бот: История сброшена.')
                continue
            try:
                response = self.chain_with_history.invoke(
                    {'question': user_text},
                    {'configurable': {'session_id': session_id}}
                )
                logging.info(f"Bot: {response.content}")
            except openai.APITimeoutError as e:
                logging.error(f"Error: {e}")
                print(f'Бот: Превышено время ожидания ответа от модели: {e}')
                continue
            except openai.APIConnectionError as e:
                logging.error(f"Error: {e}")
                print(f'Бот: Ошибка соединения с API: {e}')
                continue
            except openai.AuthenticationError as e:
                logging.error(f"Error: {e}")
                print(f'Бот: Ошибка аутентификации: {e}')
                break
            except Exception as e:
                logging.error(f"Error: {e}")
                print(f'Бот: Произошла ошибка: {e}')
                continue
            bot_reply = response.content.strip()
            print(f'Бот: {bot_reply}')


# Запуск бота
if __name__ == "__main__":
    logging.info("=== New session ===")
    model = os.getenv("OPENAI_API_MODEL", "gpt-4o")
    model_name=model, 

    bot = CliBot(
        model_name=model,
        )

        
    bot('user_123') # можно заменить на любой идентификатор сессии