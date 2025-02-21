import json
import os

from models import _constants
from controllers.rag import _re_write_query


# Load history
def load_history(user_id, collection_id, chatbot_name):
    if chatbot_name == _constants.NAME_CHATBOT_STAVIAN_GROUP:
        _path = _constants.DATAS_STAVIAN_GROUP_PATH
    else:
        _path = _constants.DATAS_PATH

    user_history_file = os.path.join(
        _path, chatbot_name, f"history_{user_id}_{collection_id}.json"
    )
    if not os.path.exists(user_history_file):
        return []

    if os.path.getsize(user_history_file) == 0:
        return []

    with open(user_history_file, "r", encoding="utf-8") as file:
        try:
            history = json.load(file)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {user_history_file}")
            return []

    return history
