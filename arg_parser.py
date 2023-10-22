import argparse

def parse_args():
    """
    Parses the command-line arguments for the script.

    Returns:
    argparse.Namespace: The namespace containing the script arguments.
    """
    desc = "A Dynamic Hugging Face Diffusers Pipeline"

    parser = argparse.ArgumentParser(description=desc)

    # Adding the script arguments with default values and help text
    parser.add_argument(
        '--config_file', type=str, default='config.json', 
        help='The directory for input data')
    

    return parser.parse_args()
