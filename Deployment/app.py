import torch
import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "."

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

id_to_label = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E"
}


@spaces.GPU
def predict(question, a, b, c, d, e):

    text = (
        f"Question: {question}\n\n"
        f"Option A: {a}\n"
        f"Option B: {b}\n"
        f"Option C: {c}\n"
        f"Option D: {d}\n"
        f"Option E: {e}"
    )

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0]

    values, indices = torch.topk(probs, 3)

    prediction = id_to_label[indices[0].item()]

    top3 = ""

    for idx, score in zip(indices, values):
        top3 += f"{id_to_label[idx.item()]} : {score.item()*100:.2f}%\n"

    return prediction, top3


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=4, label="Question"),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E"),
    ],
    outputs=[
        gr.Textbox(label="Predicted Answer"),
        gr.Textbox(label="Top 3 Predictions"),
    ],
    title="SmartMCQ Solver",
    description="Fine-tuned DeBERTa-v3-small for Multiple Choice Question Answering",
)

demo.queue()
demo.launch()
