import subprocess
# from fastapi import Form, File, UploadFile
from typing import Optional
from pathlib import Path
# from fastapi.responses import JSONResponse
import aiofiles
import asyncio
import json
from models import _environments, _prompts, _constants
from controllers.rag import _rag_qdrant, _history

async def upload_files_absolute(
    # user_id: str = Form(...),
    # file: UploadFile = File(...),
    # language: str = Form(...),
    # chatbot_name: str = Form(...),
    # exactly: Optional[int] = 0
    user_id,
    file,
    language,
    chatbot_name,
    exactly: Optional[int] = 0
):
    print("Upload_files_absolute | Exactly: ", exactly)
    if chatbot_name == _constants.NAME_CHATBOT_STAVIAN_GROUP:
        _path = _constants.DATAS_STAVIAN_GROUP_PATH
    else:
        _path = _constants.DATAS_PATH

    path = _path + "/" + chatbot_name
    user_dir = Path(path) / user_id
    # user_dir:  files\datas\user\STAVIAN_GROUP_CHAT\10
    print('user_dir: ', user_dir)
    file_path = user_dir / "Stavian_Group.pdf"
    print(file_path)

    try:
        if chatbot_name == _constants.NAME_CHATBOT_STAVIAN_GROUP:
            _path = _constants.DATAS_STAVIAN_GROUP_PATH
        else:
            _path = _constants.DATAS_PATH

        path = _path + "/" + chatbot_name
        contents = await file.read()

        file_size = len(contents)
        file_chunk_size = 1
        # print('file_size: ', file_size)
        # print('contents: ', contents)

        if file_size > _constants.MAX_FILE_SIZE:
            file_chunk_size = int(file_size // _constants.MAX_FILE_SIZE + 1)

        await file.seek(0)

        # Create user directory if it does not exist
        user_dir = Path(path) / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        file_path = user_dir / "Stavian_Group.pdf"
        print('file_path: ------------', file_path)
        # file_path: ------------ files\datas\user\STAVIAN_GROUP_CHAT\10\Stavian_Group.pdf

        # Save the uploaded file to disk in chunks
        chunk_size = 1024 * 1024  # 1MB
        # print('file_path: ', file_path)
        async with aiofiles.open(file_path, "wb") as f:
            for i in range(0, len(contents), chunk_size):
                await f.write(contents[i : i + chunk_size])
                await asyncio.sleep(0.01)  # Thêm thời gian nghỉ nhỏ

    #     # Process the uploaded file using an absolute file path
    #     # absolute_file_path = str(file_path.resolve())
    #     # command = f"python process_file.py {absolute_file_path} {user_id} {language} {chatbot_name} {file_chunk_size} {exactly}"
    #     # process = subprocess.run(command, shell=True, capture_output=True, text=True)
    #     # if process.returncode != 0:
    #     #     raise Exception(f"Command failed: {process.stderr}")
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            _rag_qdrant.save_vector_db,
            str(file_path),
            user_id,
            language,
            chatbot_name,
            file_chunk_size,
            exactly,
        )
        # print('answer: ', answer)

    #     link = (
    #         _path
    #         + "/"
    #         + chatbot_name
    #         + "/"
    #         + user_id
    #         + "/"
    #         + f"list_collections_qdrant.json"
    #     )
    #     async with aiofiles.open(link, "r", encoding="utf-8") as f:
    #         file_content = json.loads(await f.read())

    #     # content = {
    #     #     "status": 200,
    #     #     "file_content": file_content,
    #     #     "message": "File uploaded and processed successfully.",
    #     # }

    #     # return JSONResponse(content=content, status_code=200)

    except Exception as e:
        print("Error uploading files: ", e)

    # filename = "Stavian_Group.pdf"
    # file = await aiofiles.open("Stavian_Group.pdf", "rb")
    # await upload_files_absolute(
    #     filename,
    #     "language",
    #     file,
    #     "language",
    #     "chatbot_name",
    #     0
    # )

async def main():
    user_id = "10"
    language = "Việt nam"
    chatbot_name = "STAVIAN_GROUP_CHAT"

    file = await aiofiles.open("Stavian_Group.pdf", "rb")
    await upload_files_absolute(
        user_id,
        file,
        language,
        chatbot_name,
        0
    )

if __name__ == "__main__":
    asyncio.run(main())
