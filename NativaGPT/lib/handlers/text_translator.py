from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name_or_path = "tencent/Hunyuan-MT-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here
messages = [
    {"role": "user", "content": "Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house."},
]

while True:
  user_input = input("User: ")
  if user_input.lower() in ["exit", "quit"]:
      break
  messages.append({"role": "user", "content": user_input})

  tokenized_chat = tokenizer.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=False,
      return_tensors="pt"
  )

  outputs = model.generate(tokenized_chat.to(model.device), max_new_tokens=2048)
  output_text = tokenizer.decode(outputs[0])
  print("Model:", output_text)
