from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

#gpt-2
def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    p_list = probabilities.numpy().tolist()
    
    return jsonify({'probabilities': p_list})

@app.route('/attention', methods=['POST'])
def attention():
    data = request.json
    text = data.get('text', '')
    
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = [attn_layer.numpy().tolist() for attn_layer in outputs.attentions]
    return jsonify({'attentions': attentions})

if __name__ == '__main__':
    app.run(debug=True)