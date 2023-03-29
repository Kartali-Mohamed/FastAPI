import torch
from fastapi import FastAPI, Request
import uvicorn
import transformers
from transformers import AutoModel, AutoTokenizer
import mysql.connector

dbcon = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="pythondb")

mycursor = dbcon.cursor()
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "test"}

def get_model():
    tokenizer = AutoTokenizer.from_pretrained("bounedjarr/sgpt-finetuned-natcat")
    model = AutoModel.from_pretrained("bounedjarr/sgpt-finetuned-natcat")
    return tokenizer,model

tokenizer, model = get_model()


SPECB_QUE_BOS = tokenizer.encode("[", add_special_tokens=False)[0]
SPECB_QUE_EOS = tokenizer.encode("]", add_special_tokens=False)[0]

SPECB_DOC_BOS = tokenizer.encode("{", add_special_tokens=False)[0]
SPECB_DOC_EOS = tokenizer.encode("}", add_special_tokens=False)[0]

def tokenize_with_specb(texts, is_query):
    # Tokenize without padding
    batch_tokens = tokenizer(texts, padding=False, truncation=True)   
    # Add special brackets & pay attention to them
    for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
        if is_query:
            seq.insert(0, SPECB_QUE_BOS)
            seq.append(SPECB_QUE_EOS)
        else:
            seq.insert(0, SPECB_DOC_BOS)
            seq.append(SPECB_DOC_EOS)
        att.insert(0, 1)
        att.append(1)
    # Add padding
    batch_tokens = tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
    return batch_tokens

def get_weightedmean_embedding(batch_tokens, model):
    # Get the embeddings
    with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

    # Get weights of shape [bs, seq_len, hid_dim]
    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(last_hidden_state.device)
    )

    # Get attn mask of shape [bs, seq_len, hid_dim]
    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask

    return embeddings

@app.post("/indexing/doc")
async def get_embedding(text: str):
    doc = []
    doc.append(text)
    embedding = get_weightedmean_embedding(tokenize_with_specb(doc, is_query=False), model)
    doc_embedding = '''{}'''.format(embedding)
    sql = "INSERT INTO `embedding`(`text`, `embedding`) VALUES (%s, JSON_OBJECT(%s, %s))"
    val = (text, "embedding", doc_embedding)
    mycursor.execute(sql,val)
    dbcon.commit()
    return {"status" : "success"}

@app.get("/indexing/doc/{myid}")
async def get_embedding_by_id(myid:int):
    sql = "SELECT JSON_VALUE(`embedding`, '$.embedding') AS embedding FROM `embedding` WHERE `id` = %s"
    val = (myid,)
    mycursor.execute(sql,val)
    data = mycursor.fetchone()
    return data[0]

@app.post("/indexing/querie")
def get_embedding(text: str):
    querie = []
    querie.append(text)
    embedding = get_weightedmean_embedding(tokenize_with_specb(querie, is_query=True), model)
    return {"querie_embedding" : '''{}'''.format(embedding)}

if __name__ == "__main__":
    uvicorn.run("testindexing:app", host='127.0.0.1', port=8000, reload=True)