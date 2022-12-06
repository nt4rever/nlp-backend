import traceback
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from main import search as search_similar_question
from main import calc as calc_similar
from pydantic import BaseModel
from typing import List
from model import Model, get_model

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class KmeansRequest(BaseModel):
    corpus: List[str]
    n_clusters: int


class KmeansResponse(BaseModel):
    plot_json: dict


class CalcRequest(BaseModel):
    ques_1: str
    ques_2: str


@app.get("/search")
async def search(question: str, num: int):
    res = search_similar_question(question, num)
    return res


@app.post("/calc")
async def calc(ques: CalcRequest):
    res = calc_similar(ques.ques_1, ques.ques_2)
    return {"score": float(res)}


@app.post("/cluster")
async def cluster(request: KmeansRequest, model: Model = Depends(get_model)):
    if (request.n_clusters < 1):
        raise HTTPException(status_code=400,
                            detail="Number of clusters must be greater than 0.")
    elif (len(request.corpus) < 2):
        raise HTTPException(status_code=400,
                            detail="Corpus must have at least 2 sentences.")
    elif (len(request.corpus) < request.n_clusters):
        raise HTTPException(status_code=400,
                            detail="Number of sentences must be greater than or equal to number of clusters.")
    try:
        result = plot_html = model.fit_kmeans(
            request.corpus, request.n_clusters)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=traceback.format_exc(),
        )
    return result
