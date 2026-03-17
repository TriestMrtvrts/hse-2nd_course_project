from aiogram import F, Router
from aiogram import Bot, Dispatcher, html
from aiogram.filters import Command
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.fsm.state import StatesGroup, State
from aiogram import types
from aiogram.types import FSInputFile

import sqlite3
import hashlib
import logging
import os
import random
from pathlib import Path
from textwrap import dedent
from functools import wraps

from keyboards.kb import auth_kb, askq_kb

from RAG_CODE.rag.rag_inference import EMBEDDING_MODEL, LOG_DB_NAME, EMBEDDING_MODEL, LLM_MODEL
from RAG_CODE.rag.rag_inference import prompt_lia, client, \
    load_question_pool, setup_database, setup_log_database, retrieve_context, retrieve_matches, \
    call_model, ask_question, ask_question_creative

from bot import db, embedding_model, conversation_history, log_db, chat_id
from bot import bot


router = Router()
QUESTION_POOL_PATH = Path(__file__).resolve().parent.parent / "experiments" / "question_pool.csv"


def _extract_document_paths(meta_datas, sources):
    candidates = []
    for value in [*meta_datas, *sources]:
        if isinstance(value, str) and os.path.exists(value):
            candidates.append(value)

    unique_paths = []
    for path in candidates:
        if path not in unique_paths:
            unique_paths.append(path)
    return unique_paths


def _format_source_names(paths):
    if not paths:
        return "Не удалось определить файл-источник."
    return ", ".join(os.path.basename(path) for path in paths)


async def _send_source_files(message: Message, paths):
    for path in paths:
        try:
            await message.answer_document(
                FSInputFile(path),
                caption=f"Файл-источник: {os.path.basename(path)}",
            )
        except Exception as exc:
            logging.warning("Failed to send source file %s: %s", path, exc)


def _lookup_question_by_id(question_id: str):
    if not QUESTION_POOL_PATH.exists():
        return None
    for row in load_question_pool(str(QUESTION_POOL_PATH)):
        if row.get("id") == question_id:
            return row
    return None


def _format_eval_matches(matches):
    if not matches:
        return "Ничего не найдено в top-k."

    lines = []
    for index, match in enumerate(matches[:5], start=1):
        snippet = " ".join(match["text"].split())[:180]
        lines.append(
            f"{index}. {match['document_name'] or 'Неизвестный документ'} | "
            f"section: {match['section'] or 'без заголовка'} | "
            f"distance: {match['distance']:.4f}\n"
            f"   {snippet}"
        )
    return "\n".join(lines)


async def _send_eval_response(message: Message, raw_query: str):
    question_row = _lookup_question_by_id(raw_query) if raw_query.isdigit() else None
    query = question_row["question"] if question_row else raw_query
    matches = retrieve_matches(query, db, embedding_model, k=5)
    source_paths = _extract_document_paths(
        [match["section"] for match in matches],
        [match["source"] for match in matches],
    )

    expected_document = question_row["expected_document"] if question_row else "не задан"
    eval_text = (
        f"Вопрос: {query}\n"
        f"Ожидаемый документ: {expected_document}\n\n"
        f"Top-k retrieval:\n{_format_eval_matches(matches)}"
    )
    await message.answer(eval_text, reply_markup=askq_kb())
    await _send_source_files(message, source_paths[:3])

# Определяем состояния для FSM 
class StateMachine(StatesGroup):
    something_like_login = State() #test
    settings = State() #think about it
    waiting_for_question = State()
    waiting_for_question_creative = State()
    waiting_for_eval = State()
    waiting_for_file = State()

    waiting_for_email = State()
    waiting_for_password = State()

# Определяем состояния для FSM процесса регистрации пользователя
class RegisterUser(StatesGroup):
    waiting_for_email = State()
    waiting_for_password = State()
    waiting_for_access_level = State()
    waiting_for_region = State()
    waiting_for_department = State()
    waiting_for_position = State()


# Функция для получения уровня доступа по user_id
def get_user_access_level(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''SELECT access_level FROM users WHERE userID = ?''', (user_id,))
    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]  # Возвращаем уровень доступа
    return None  # Если пользователь не найден или разлогинен

# Декоратор для проверки уровня доступа
def access_level_required(level):
    def decorator(func):
        @wraps(func)
        async def wrapped(message: types.Message, *args, **kwargs):
            user_id = message.from_user.id  # Получаем user_id пользователя
            
            # Проверяем уровень доступа по user_id
            access_level = get_user_access_level(user_id)
            
            if access_level is None:
                await message.answer("Вы не авторизованы. Пожалуйста, войдите в систему.")
                return

            if access_level >= level:
                return await func(message, *args, **kwargs)
            else:
                await message.answer("У вас недостаточно прав для выполнения этой команды.")
        return wrapped
    return decorator


# @router.message(Command(commands=['upload_file']))
@router.message(F.content_type.in_({'document'}))
@access_level_required(3)
async def file_handler(message: Message):
    """
    This handler receives document sent by the user and saves them to a local directory.
    """
    try:
        document = message.document

        file = await bot.get_file(document.file_id)
        file_path = file.file_path

        save_directory = 'RAG_CODE/rag/data'
        os.makedirs(save_directory, exist_ok=True)  # Ensure directory exists

        # Define the path
        # global file_name
        file_name = document.file_name
        destination = os.path.join(save_directory, file_name)

        await bot.download_file(file_path, destination)

        # Sends TestLogMessage to user, delete in prod.
        await message.answer(f"Файл '{file_name}' был успешно сохранён!")

    except Exception as exctact_e:
        logging.error("Error while downloading file")
        await message.answer('Возникла ошибка при добавлении файла') # 🥴🥴🥴


# Функция для связывания user_id с email
def save_user_id_to_db(user_id, email):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''UPDATE users SET userID = ? WHERE email = ?''', (user_id, email))
    conn.commit()
    conn.close()

# Функция для добавления пользователя в базу данных
def add_user_to_db(email, password, access_level, region, department, position):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    password_hash = generate_password_hash(password)
    
    try:
        cursor.execute('''INSERT INTO users (email, password, access_level, region, department, position) VALUES (?, ?, ?, ?, ?, ?)''',
                       (email, password_hash, access_level, region, department, position))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Хендлер команды для регистрации нового пользователя (доступна только администраторам)
@router.message(Command(commands=['register_user']))
@access_level_required(3)
async def start_register_new_user(message: types.Message, state: FSMContext):
    await message.answer("Введите email нового пользователя.")
    
    # Переводим в состояние ожидания email
    await state.set_state(RegisterUser.waiting_for_email)

# Хендлер для получения email
@router.message(RegisterUser.waiting_for_email)
async def process_email(message: types.Message, state: FSMContext):
    email = message.text
    await state.update_data(email=email)
    
    await message.answer("Теперь введите пароль нового пользователя.")
    
    # Переводим в состояние ожидания пароля
    await state.set_state(RegisterUser.waiting_for_password)


# Хендлер для получения пароля
@router.message(RegisterUser.waiting_for_password)
async def process_password(message: types.Message, state: FSMContext):
    password = message.text
    await state.update_data(password=password)
    
    await message.answer("Введите регион нового пользователя (Например: Крайний север):")
    
    # Переводим в состояние ожидания региона
    await state.set_state(RegisterUser.waiting_for_region)


# Хендлер для получения региона
@router.message(RegisterUser.waiting_for_region)
async def process_region(message: types.Message, state: FSMContext):
    region = message.text
    await state.update_data(region=region)
    
    await message.answer("Введите департамент нового пользователя:")
    
    # Переводим в состояние ожидания департамента
    await state.set_state(RegisterUser.waiting_for_department)

# Хендлер для получения департамента
@router.message(RegisterUser.waiting_for_department)
async def process_region(message: types.Message, state: FSMContext):
    department = message.text
    await state.update_data(department=department)
    
    await message.answer("Введите должность нового пользователя:")
    
    # Переводим в состояние ожидания должности
    await state.set_state(RegisterUser.waiting_for_position)

# Хендлер для получения должности
@router.message(RegisterUser.waiting_for_position)
async def process_region(message: types.Message, state: FSMContext):
    position = message.text
    await state.update_data(position=position)
    
    await message.answer("Введите уровень доступа для нового пользователя (1 — обычный пользователь, 2 - менеджер, 3 — администратор).")
    
    # Переводим в состояние ожидания уровня доступа
    await state.set_state(RegisterUser.waiting_for_access_level)


# Хендлер для получения уровня доступа
@router.message(RegisterUser.waiting_for_access_level)
async def process_access_level(message: types.Message, state: FSMContext):
    access_level = message.text

    # Проверяем, что введён корректный уровень доступа
    if not access_level.isdigit() or int(access_level) not in [0, 1, 2, 3]:
        await message.answer("Некорректный уровень доступа. Введите число от 0 до 3")
        return

    access_level = int(access_level)

    # Получаем все данные из FSM
    user_data = await state.get_data()
    email = user_data['email']
    password = user_data['password']
    region = user_data['region']
    department = user_data['department']
    position = user_data['position']

    # Добавляем нового пользователя в базу данных
    if add_user_to_db(email, password, access_level, region, department, position):
        await message.answer(f"Пользователь {email} успешно зарегистрирован с уровнем доступа {access_level}.")
    else:
        await message.answer("Ошибка: пользователь с таким email уже существует.")

    await state.clear()
    
# Функция для разлогинивания пользователя (удаление userID)
def logout_user(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Обнуляем поле userID, разлогинивая пользователя
    cursor.execute(f'''UPDATE users SET userID = {random.randint(1, 1000000)} WHERE userID = ?''', (user_id,))
    conn.commit()
    conn.close()

# Хендлер для команды logout
@router.message(Command(commands=['logout']))
async def handle_logout(message: types.Message, state: FSMContext):
    await state.clear()

    await message.answer("Сессия сброшена. Нажмите '▶️ Начать', чтобы снова открыть главное меню.", reply_markup=auth_kb())


# Функция для запроса статистики (доступна только администраторам)
@router.message(Command(commands=['get_statistics']))
@access_level_required(3)
async def get_statistics(message: types.Message, state: FSMContext):
    # Логика запроса статистики (например, количество пользователей)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''SELECT COUNT(*) FROM users''')
    total_users = cursor.fetchone()[0]

    cursor.execute('''SELECT COUNT(*) FROM users WHERE access_level = 3''')
    total_admins = cursor.fetchone()[0]

    conn.close()

    await message.answer(f"Всего пользователей: {total_users}\nИз них администраторов: {total_admins}")
    await message.answer(f"Готовим анализ статистики запросов пользователей..")

    conn = sqlite3.connect('/Users/maximmashtaler/Projects/prog/hacks/CP/szfo2024/cp11-10/log.sqlite3')
    cursor = conn.cursor()

    cursor.execute('''SELECT question FROM logs''')
    total_questions = cursor.fetchall()

    print(total_questions)

    query = 'чем чаще всего интересуются пользователи?'

    prompt = dedent(f"""
    Используя историю вопросов пользователей:

    ```
    {total_questions}
    ```

    развернуто ответь как аналитик на вопрос администратора:
    {query}
    """)
    

    response = call_model(prompt, conversation_history, temp=0.2)

    await message.answer(response, reply_markup=askq_kb())
    await state.clear()



# @router.message(Command(commands=['get_stat']))
# @access_level_required(3)
# async def stat_answer(message: Message, state: FSMContext):
#     '''
#     Функция для вопросов по статистике использования сервиса (для админов)
#     '''

#     conn = sqlite3.connect('/Users/maximmashtaler/Projects/prog/hacks/CP/szfo2024/cp11-10/log.sqlite3')
#     cursor = conn.cursor()

#     cursor.execute('''SELECT question FROM logs''')
#     total_questions = cursor.fetchall()

#     print(total_questions)

#     query = 'чем чаще всего интересуются пользователи?'

#     prompt = dedent(f"""
#     Используя историю вопросов пользователей:

#     ```
#     {total_questions}
#     ```

#     ответь на вопрос администратора:
#     {query}
#     """)
    

#     response = call_model(prompt, conversation_history, temp=0.2)

#     await message.answer(response, reply_markup=askq_kb())
#     await state.clear()

    




@router.message(Command(commands=["start"]))
async def command_start_handler(message: Message, state: FSMContext) -> None:
    """
    Sends Hello message to User
    """
    await state.clear()
    await message.answer(
        f"Привет, {html.bold(message.from_user.full_name)}! Добро пожаловать в нашего бота. Нажмите '▶️ Начать', чтобы перейти в главное меню.",
        reply_markup=auth_kb(),
    )

@router.message(F.text.in_({'▶️ Начать', '🔐 Авторизация'}))
async def auth_user(message: Message, state: FSMContext):
    '''
    Функция для входа в главное меню без авторизации
    '''
    await state.clear()
    await message.answer("Главное меню открыто. Можете сразу задавать вопрос.", reply_markup=askq_kb())


# Функция для генерации хеша пароля
def generate_password_hash(password):
    '''
    Функция для генерации хеша пароля
    '''
    return hashlib.sha256(password.encode()).hexdigest()

# Функция для проверки логина и пароля
def authenticate_user(email, password):
    '''
    Функция для проверки логина и пароля
    '''
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    cursor.execute('''SELECT password FROM users WHERE email = ?''', (email,))
    user = cursor.fetchone()
    conn.close()

    if user:
        stored_password = user[0]
        if (stored_password) == generate_password_hash(password):
            return True
    return False




@router.message(F.text == '⚙️ Настройки')
async def settings(message: Message):
    ... #logic for settings.
    await message.reply('Скоро тут будут настройки!')


@router.message(Command(commands=["eval"]))
async def eval_retrieval(message: Message):
    raw_query = message.text.partition(" ")[2].strip()
    if not raw_query:
        await message.answer("Используйте /eval <вопрос> или /eval <id_из_question_pool.csv>")
        return
    await _send_eval_response(message, raw_query)


@router.message(F.text == '🧪 Eval retrieval')
async def ask_eval_query(message: Message, state: FSMContext):
    await message.reply(
        'Введите вопрос для retrieval-проверки или id из experiments/question_pool.csv:',
        reply_markup=askq_kb(),
    )
    await state.set_state(StateMachine.waiting_for_eval)


@router.message(StateMachine.waiting_for_eval)
async def run_eval_query(message: Message, state: FSMContext):
    await _send_eval_response(message, message.text.strip())
    await state.clear()


@router.message(F.text == '📄 Задать вопрос')
async def ask_q(message: Message, state: FSMContext):
    '''
    Функция чтобы задать вопрос
    '''
    await message.reply('Введите ваш вопрос:', reply_markup=askq_kb())

    await state.set_state(StateMachine.waiting_for_question)


@router.message(StateMachine.waiting_for_question)
async def answer(message: Message, state: FSMContext):
    '''
    Функция для ответа на вопрос в режиме QnA
    '''
    await message.reply(f'Ваш вопрос: {message.text}', reply_markup=askq_kb())
    response, context, meta_datas, sources = ask_question(message.text, db, embedding_model, conversation_history, log_db, chat_id)
    source_paths = _extract_document_paths(meta_datas, sources)
    source_names = _format_source_names(source_paths)
    source_sections = [item for item in set([*meta_datas, *sources]) if item not in source_paths]
    sections_text = ', '.join(source_sections) if source_sections else "Не удалось определить раздел."
    ans = f"""
    \nОтвет:
    {response}
    \nОткуда взята информация:
    {source_names}
    \nИсточники:
    {sections_text}
    """
    await message.answer(ans, reply_markup=askq_kb())
    await _send_source_files(message, source_paths)
    await state.clear()



@router.message(F.text == '💡 Задать вопрос в креативном режиме')
async def ask_q_creative(message: Message, state: FSMContext):
    '''
    Функция чтобы задать вопрос
    '''
    await message.reply('Введите ваш вопрос:', reply_markup=askq_kb())

    await state.set_state(StateMachine.waiting_for_question_creative)


@router.message(StateMachine.waiting_for_question_creative)
async def creative_answer(message: Message, state: FSMContext):
    '''
    Функция для ответа на вопрос в креативном режиме
    '''
    await message.reply(f'[CREATIVE] Ваш вопрос: {message.text}', reply_markup=askq_kb())
    response, context, meta_datas, sources = ask_question_creative(message.text, db, embedding_model, conversation_history, log_db, chat_id)
    source_paths = _extract_document_paths(meta_datas, sources)
    source_names = _format_source_names(source_paths)
    source_sections = [item for item in set([*meta_datas, *sources]) if item not in source_paths]
    sections_text = ', '.join(source_sections) if source_sections else "Не удалось определить раздел."
    ans = f"""
    \nОтвет:
    {response}
    \nОткуда взята информация:
    {source_names}
    \nИсточники:
    {sections_text}
    """
    await message.answer(ans, reply_markup=askq_kb())
    await _send_source_files(message, source_paths)
    await state.clear()

    
