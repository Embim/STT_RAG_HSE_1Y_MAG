import asyncio

from settings import settings
from system.rag.vectore_store import VectorStoreManager
from system.rag.pipeline import run

query = 'Состав красной армии в 1941 году'


async def main():
    answer= await run(question=query)

    return answer

if __name__=='__main__':
    print(asyncio.run(main()))