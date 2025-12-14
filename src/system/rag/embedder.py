from settings import settings
import requests

class LocalEmbedder:
    def __init__(self):
        self.url = settings.EMBEDDING_URL
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.url,
            json={"input": texts}
        )
        data = response.json()
        return [item["embedding"] for item in data["data"]]
    
    def embed_query(self, text: str) -> list[float]:
        response = requests.post(
            self.url,
            json={"input": text}
        )
        data = response.json()
        return data["data"][0]["embedding"]