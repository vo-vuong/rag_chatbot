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