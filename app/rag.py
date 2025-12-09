# 벡터 db 로드 + 검색기능
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import FAISS_INDEX_PATH, METADATA_PATH, EMBED_MODEL

# 임베딩 모델 로드
embed_model = SentenceTransformer(EMBED_MODEL) # query 임베딩을 위한 것인가?!!

# 벡터 db 로드
index = faiss.read_index(FAISS_INDEX_PATH) # vecotr db에 저장된 유사도 측정 준비가 완료된 벡터와된 rag 데이터들

# temp_chunksa 로드 -> 처음엔 진짜 메타데이터인줄 알았으나 그 메타데이터가 아니라 chunk 적용한 원본데이터를 말함
temp_chunks = []
with open(METADATA_PATH,'r') as f:
    for line in f:
        temp_chunks.append(json.loads(line))
    
def embed_text(text:str):
    vec = embed_model.encode([text]).astype('float32') # text 받는게 없는 뎁쇼? 뒤에 생기려나..? 생겨야 할텐데..ㅋ query를 전달 받는 거면 인정
    return vec
    
def search(query:str, k:int = 5): # 여기서 k는 무슨 역할을 하길래 이러지?
    qvec = embed_text(query)
    distances, ids = index.search(qvec,k) # query를 임베딩 한 거랑, k랑 같이 전달하는데 k가 도통 무슨 역할인지는 모르겠다ㅏㅏ
    
    results = []
    for idx in ids[0]:
        results.append(temp_chunks[idx]['chunk'])
        
    return results