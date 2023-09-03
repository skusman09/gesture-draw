from src.controller import AppController


# Air Canvas application entrypoint.
def main() -> None:
    app = AppController()
    app.run()


if __name__ == "__main__":
    main()
