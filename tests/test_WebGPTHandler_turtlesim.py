from NativaGPT import *
from WebGPTHandler import ChatGPTInteractionManager

if __name__ == '__main__':

    # Initialize ChatGPT handler and NativaGPT API
    gpt_handler = ChatGPTInteractionManager()
    api = NativaGPTApi(api_key="YOUR_API_KEY_HERE")

    initial_prompt = (
        "Here you have a JSON config file that contains functions to interact with Turtlesim from ROS Noetic. "
        "You can call each function by sending the exact JSON string. The result will be returned to you.\n"
        "To call a function, please use the entire function string, as described in the JSON file.\n"
        "Please always use the functions within the JSON file to fulfill my requests, and call them by providing the full definition with modified parameters in a JSON string.\n"
        "After every function you send, I will automatically return the output, and I want you to comment on it briefly.\n"
        "Thanks, ChatGPT."
    )

    try:
        # Launch the ChatGPT interaction page
        gpt_handler.launch_chatpgt_page()

        # Perform login
        gpt_handler.login()

        # Load the configuration file containing function definitions
        gpt_handler.input_external_file("../config/turtlesim_functions.json")

        # Send initial prompt and retrieve response
        response = gpt_handler.send_and_receive(initial_prompt)

        cycle = 0
        while True:
            print(f"Cycle {cycle}")

            # Get user input
            user_prompt = input("Prompt: ")
            if user_prompt.lower() in ['exit', 'quit']:  # Exit condition
                print("Exiting...")
                break

            response = gpt_handler.send_and_receive(user_prompt)

            # Execute the JSON string received in the response
            functions_output, text = api.run(response=response)

            # Send function output back to ChatGPT
            gpt_handler.send_and_receive(functions_output)

            cycle += 1

    except Exception as e:
        print(f"An error occurred: {e}")