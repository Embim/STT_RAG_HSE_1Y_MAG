import asyncio

from settings import settings
from system.rag.vectore_store import VectorStoreManager
from system.rag.pipeline import run

query = 'что такое Фотон'


async def main():
    answer, context = await run(question=query)

    return context, '\n', answer

if __name__=='__main__':
    print(asyncio.run(main()))