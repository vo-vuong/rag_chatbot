from models import _prompts, _environments
from langchain_core.output_parsers import StrOutputParser


def rewrite_query(query, prompt):
    chain = (
        prompt | _environments.get_llm("gpt-4o-mini", temperature=0) | StrOutputParser()
    )
    results = chain.invoke({"input": str(query)})

    return results