from fastapi import Body, HTTPException
from typing import Optional



from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate


from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()

def chat(query: str = Body("", description="User's query"),
         model_name: str = Body("glm4", description="Base model name"),
         temperature: float = Body(0.8, description="temperature", ge=0.0, le=2.0),
         max_tokens: Optional[int] = Body(None, description="max_tokens"),
         ):


    logger.info("Received query: {}", query)
    logger.info("Model name: {}", model_name)
    logger.info("Temperature: {}", temperature)
    logger.info("Max tokens: {}", max_tokens)

    # LangChain call glm4
    try:
        # use LangChain for glm4
        template = """{query}"""
        prompt = PromptTemplate.from_template(template)


        api_key=os.environ.get('ZHIPU_API_KEY')
        
        llm = ChatZhipuAI(
            zhipuai_api_key = api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        llm_chain = prompt | llm
        response = llm_chain.invoke(query)

        if response is None:
            raise ValueError("Received null response from LLM")

        return {"LLM Response": response}

    except ValueError as ve:
        # Bad Request for invalid input
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Internal Server Error for other exceptions
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
