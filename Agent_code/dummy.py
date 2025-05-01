from main import main as gen_response

def main(user_input: str) -> str:
    
    response = gen_response(user_input + " to you, too.")
    return response

if __name__ == "__main__":
    main()