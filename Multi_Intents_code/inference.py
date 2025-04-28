from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "/content/drive/MyDrive/EBA5004/t5-multi-intentv2/t5-multi-intent-v3/checkpoint-225000"  
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

#original text inputs
input_text = "multi-intent: What is the deadline for MATH204 assignment? Where do I upload it?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# inference
outputs = model.generate(input_ids, max_length=128)
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Predicted:", decoded_output)