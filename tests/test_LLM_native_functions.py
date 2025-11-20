from NativaGPT import NativaGPTApi, LLMPromptHandler
from NativaGPT import LLMPromptHandler as llm_prompt
from WebGPTHandler import ChatGPTInteractionManager

if __name__ == '__main__':

    api = NativaGPTApi(api_key="aaa")

    initial_prompt = ("Here you have an json config file that has some native functions of my Operative System. You can call each one of this functions by sending the exact JSON string. The result will be then returned to you. \n"
                      "To call a function please use the whole function string, as described in the JSON file. \n"
                      "Please, to fulfill my requests, please use ALWAYS the functions inside the JSON files I give to you. \n"
                      "After every function you send, I will send you the output back automatically, and I want you to comment it briefly. \n"
                      "Thanks.")

    # Create prompt module
    llm_prompt_module = llm_prompt(config="../config/config_default.json")

    #Initiate a conversation
    llm_prompt_module.send_to_llm(content=initial_prompt, file_path="../config/functions/native_functions.json")

    i = 0
    while True:

        print(f"Cycle {i}")

        user_input = input("Prompt:")

        # Ask user for input
        response = llm_prompt_module.send_to_llm(content=user_input, file_path="../config/functions/native_functions.json")

        print(response)

        # Run the required functions
        functions_output, text = api.run(response=response)

        # Send command output to LLM
        #llm_prompt_module.send_to_llm(temperature=0.0, content=functions_output, file_path="../config/native_functions.json")

        print(text)
        i += 1