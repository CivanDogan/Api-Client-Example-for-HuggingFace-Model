import concurrent
import json
import requests
import asyncio

API_KEY = ""
headers = {"Authorization": f"Bearer {API_KEY}"}
API_URL = "https://api-inference.huggingface.co/models/deprem-ml/deprem-ner"


def query(payload):
    data = json.dumps(payload)

    # make request async
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

with open('data.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
    # get text
    tweets = [x["_source"]["text"] for x in data['hits']['hits']]

import time

async def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor,
                query,
                tweet,
            )
            for tweet in tweets
        ]

        for response in await asyncio.gather(*futures):
            print(response)


t = time.time()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

print(f"Took : {time.time() - t}")


"""
Results:
With running 3 gpus on serverside (HuggingFace)

1. 20 Workers : 41.01 second (48.5 tweets per second)
2. 50 Workers : 17.12 second (117.5 tweets per second)
2. 200 Workers : 7 second (286 tweets per second)
3. 500 Workers : 8.1 second (246 tweets per second)

Adding more workers, not reduces the run time.
"""