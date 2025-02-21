import re
import sys
import os
from langchain_core.prompts import ChatPromptTemplate
from controllers.rag import _chain_invoke
from models import _prompts


def re_write_query(query, history):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _prompts.RE_WRITE_QUERY.format(history=str(history), query=str(query)),
            ),
        ]
    )

    ans = _chain_invoke.rewrite_query(query, prompt)
    answer = (
        "CÂU TRUY VẤN GỐC: "
        + query
        + "\nCÂU TRUY VẤN ĐƯỢC LÀM RÕ (tham khảo để hiểu hơn): "
        + ans
    )

    return answer


if __name__ == "__main__":
    query = "Tôi tên gì?"
    history = "Tôi tên khỉ khô"
    print(re_write_query(query, history))
