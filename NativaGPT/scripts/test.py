from NativaGPT.scripts.nativa_mcp_wrapper import NativaMCPWrapper

def main():

  wrapper = NativaMCPWrapper(system_context="You are an industrial HMI assistant. Be brief.", config_path="/home/pedro/Documents/uv-projects/NativaGPT/config/config_default.json")

  while True:

    response = wrapper.ask(input("Prompt:"))

    print(response)

if __name__ == "__main__":

  main()
