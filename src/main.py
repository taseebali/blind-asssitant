# """
# Main entry point for the Blind Assistant application
# """

from src.core.assistant import BlindAssistant

def main():
    """Initialize and start the Blind Assistant."""
    assistant = BlindAssistant(show_display=True)
    assistant.start()

if __name__ == "__main__":
    main()