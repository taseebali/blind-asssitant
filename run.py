from src.core.assistant import BlindAssistant

def main():
    # Note: only include port in camera_ip if it's not the default 8080
    assistant = BlindAssistant(
        show_display=True,
        camera_ip="172.17.209.194"  # Just the IP address, port is handled internally
    )

    try:
        assistant.start()
    except KeyboardInterrupt:
        print("Shutting down Blind Assistant...")
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main()