from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()

client = TavilyClient()


def search_web(query: str) -> str:
    query = query.strip()

    if len(query) < 2:
        return "No search performed because the query was too short."

    response = client.search(query=query, max_results=3)

    results = []

    for r in response["results"]:
        results.append(
            f"Title: {r['title']}\n"
            f"Content: {r['content']}\n"
            f"Source: {r['url']}"
        )

    return "\n\n".join(results)