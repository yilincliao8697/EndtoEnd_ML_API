import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel, ConfigDict, Field, field_validator
from redis import asyncio
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_path = "./distilbert-base-uncased-finetuned-sst2"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    top_k=None,
)

logger = logging.getLogger(__name__)
LOCAL_REDIS_URL = "redis://localhost:6379"


@asynccontextmanager
async def lifespan_mechanism(app: FastAPI):
    HOST_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
    logger.debug(HOST_URL)
    redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache-project")

    yield


sub_application_sentiment_predict = FastAPI(lifespan=lifespan_mechanism)


class SentimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text:list[str]


class Sentiment(BaseModel):
    label: str
    score: float


class SentimentResponse(BaseModel):
    # ... [Sentiment]
    predictions: list[list[Sentiment]]


@sub_application_sentiment_predict.post(
    "/bulk-predict", response_model=SentimentResponse
)
@cache(expire=60)
async def predict(sentiments: SentimentRequest):
    return {"predictions": classifier(sentiments.text)}


@sub_application_sentiment_predict.get("/health")
async def health():
    return {"status": "healthy"}
