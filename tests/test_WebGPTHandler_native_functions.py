from NativaGPT import *
from WebGPTHandler import ChatGPTInteractionManager

if __name__ == '__main__':

    gpt_hander = ChatGPTInteractionManager()
    api = NativaGPTApi(api_key="")

    initial_prompt = ("Here you have an json config file that has some native functions of my Operative System. You can call each one of this functions by sending the exact JSON string. The result will be then returned to you. \n"
                      "To call a function please use the whole function string, as described in the JSON file. \n"
                      "Please, to fulfill my requests, please use ALWAYS the functions inside the JSON files I give to you. \n"
                      "After every function you send, I will send you the output back automatically, and I want you to comment it briefly. \n"
                      "Thanks Chatgpt.")

    # Initiate gpt page
    gpt_hander.launch_chatpgt_page()

    # Call for login
    gpt_hander.login()

    #Initiate a conversation
    gpt_hander.input_external_file("../config/native_functions.json")
    response = gpt_hander.send_and_receive(initial_prompt)
    i = 0
    while True:

        print(f"Cycle {i}")

        # Ask user for input
        gpt_hander.send_and_receive(input("Prompt: "))

        # Run the required functions
        functions_output, text = api.run(response=response)

        # Send command output to LLM
        gpt_hander.send_and_receive(functions_output)

        i += 1