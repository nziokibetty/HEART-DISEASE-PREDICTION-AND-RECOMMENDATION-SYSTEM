from pyexpat import model
import re
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import openai

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static",StaticFiles(directory="static",html=True),name="static")

model = joblib.load("model.pkl")

openai.api_key = "YOUR API KEY"

model_id = "YOUR MODEL ID"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/prediction", response_class=HTMLResponse)
async def prediction(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request, "num_inputs": range(13)})

@app.get("/app", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    response = openai.chat.completions.create(
        messages=[
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": "This is a recommender chatbot for heart disease predictions."},
        ],
        model=model_id,
        temperature=0.7,
        max_tokens=100
    )
    chat_response = response.choices[0].message.content
    formatted_response = format_response(chat_response)
    return templates.TemplateResponse("app.html", {"request": request, "user_input": user_input, "chat_response": formatted_response})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, user_input: str = Form(...)):

    features = np.array(user_input, dtype=int).reshape(1, -1) 

    prediction = model.predict(features)

    formatted_prediction = format_prediction(prediction)

    return templates.TemplateResponse("app.html", {"request": request, "user_input": user_input, "prediction": formatted_prediction})

def format_prediction(prediction):
    return prediction 


def format_response(response_text):
    lines = response_text.split('\n')
    formatted_lines = [line for line in lines if re.match(r'^\d+\.', line)]
    formatted_response = '\n'.join(formatted_lines)
    return formatted_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
