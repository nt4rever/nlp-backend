import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
file = './store/csv/java.csv'
df = pd.read_csv(file)
model = SentenceTransformer('./store/all-mpnet-base-v2')
df['EMBEDDING'] = df['TOKENIZE'].apply(model.encode)
sen_embeddings = df['EMBEDDING'].to_list()


def search(question, num=5):
    question_embedding = model.encode(question)
    score = cosine_similarity(
        [question_embedding],
        sen_embeddings
    )
    res = df[['ID', 'QUESTION', 'ANSWER']].copy()
    res['SCORE'] = score[0]
    res.sort_values(by='SCORE', inplace=True, ascending=False)
    return res.head(num).to_json(orient='records', date_format='iso') 
