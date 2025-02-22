import re
from unstructured.cleaners.core import (
    group_broken_paragraphs,
    clean,
    group_broken_paragraphs,
)


unwanted_chars = [
    "ü",
    "ÿ",
    "þ",
    "®",
    "±",
    "©",
    "µ",
    "÷",
    "v ",
    "Ø",
    "€",
    "ƒ",
    "†",
    "‡",
    "ˆ",
    "Š",
    "Œ",
    "Ä",
    "•",
    "˜",
    "™",
    "š",
    "›",
    "œ",
    "ž",
    "Ÿ",
    "¡",
    "¢",
    "£",
    "¤",
    "¥",
    "¦",
    "§",
    "¨",
    "ª",
    "«",
    "¬",
    "°",
    "²",
    "³",
    "µ",
    "¶",
    "»",
    "¿",
    "Ä",
    "ü ",
    "ÿ ",
    "þ ",
    "® ",
    "± ",
    "© ",
    "µ ",
    "÷ ",
    "v ",
    "Ø ",
    "€ ",
    "ƒ ",
    "† ",
    "‡ ",
    "ˆ ",
    "Š ",
    "Œ ",
    "Ä ",
    "• ",
    "˜ ",
    "™ ",
    "š ",
    "› ",
    "œ ",
    "ž ",
    "Ÿ ",
    "¡ ",
    "¢ ",
    "£ ",
    "¤ ",
    "¥ ",
    "¦ ",
    "§ ",
    "¨ ",
    "ª ",
    "« ",
    "¬ ",
    "° ",
    "² ",
    "³ ",
    "µ ",
    "¶ ",
    "» ",
    "¿ ",
    "Ä ",
]


# Fix mấy ngoặc nhọn với đồ trong string
def validate_and_fix_braces(text):
    text = str(text)
    # Định nghĩa các ký tự đặc biệt cần escape
    special_characters = {"{": "\\(", "}": "\\)", "]": "\\)", "[": "\\(", '"': "\\'"}
    # special_characters = {'{': '\\{', '}': '\\}', ']': '\\]', '[': '\\[', '"': '\\"', "'": "\\'"}

    # Thay thế từng ký tự đặc biệt trong chuỗi đầu vào
    for char, escape_char in special_characters.items():
        text = text.replace(char, escape_char)

    return text

def remove_specific_chars(text, chars):
    pattern = re.compile("|".join(re.escape(char) for char in chars))
    cleaned_text = pattern.sub("-", text)
    return cleaned_text

def remove_char_dots(text):
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(
        r" {3,}", " ", text
    )  # Thay thế một hoặc nhiều dòng trắng bằng một dòng mới
    text = re.sub(r"\.\.\.", ".", text)  # Thay thế "..." bằng "."
    text = re.sub(r"\.\.\.", ".", text)  # Thay thế "..." bằng "."
    text = re.sub(r"\.\.\.", ".", text)  # Thay thế "..." bằng "."

    return text

def clean_data_unstructured(texts):
    _text = []
    for text in texts:
        # print(text)
        # text = clean_text(text, unwanted_chars)
        text = remove_specific_chars(text, unwanted_chars)
        # text = clean(text, extra_whitespace=True, dashes=True, bullets=True, lowercase=False)
        text = group_broken_paragraphs(text)
        text = remove_char_dots(text)
        _text.append(text)
    return _text
