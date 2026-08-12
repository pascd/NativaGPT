"""Standalone CLI text translation script using a local HuggingFace model.

Loads the ``tencent/Hunyuan-MT-7B`` causal language model and tokenizer via
``transformers`` (``AutoModelForCausalLM``/``AutoTokenizer``) and runs an
interactive input loop: each line typed by the user is appended to the chat
history and sent to the model, which generates a translation that is
printed to stdout. Type ``exit`` or ``quit`` to stop.

This module is not imported for its side effects: model loading and the
input loop only run when the script is executed directly (``python3
text_translator.py``), guarded behind ``if __name__ == "__main__":``.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name_or_path = "tencent/Hunyuan-MT-7B"


def main():
    """Load the translation model/tokenizer and run the interactive translation loop.

    Loads the ``tencent/Hunyuan-MT-7B`` model and tokenizer, then repeatedly
    prompts for user input on stdin, appends each input to the running chat
    history, generates a model response, and prints it. Exits the loop when
    the user types ``exit`` or ``quit``.
    """
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


if __name__ == "__main__":
    main()
