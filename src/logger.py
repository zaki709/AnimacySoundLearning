import os


class Logger:
    def __init__(self, train_id: str) -> None:
        self.file_path = f"output/log/log_{train_id}.txt"
        if not os.path.exists(self.file_path):
            with open(self.file_path, "a") as log:
                log.write(f"Log: {train_id}\n\n")

    def log(self, content: str) -> None:
        with open(self.file_path, "a") as log:
            log.write(f"{content}\n")

    def log_params(self, prefix: str = "", params: dict = {}) -> None:
        lines = ""
        if prefix != "":
            lines += f"{prefix}\n"
        try:
            for key in params.keys():
                lines += f"{key}: {params[key]}\n"
        except Exception as e:
            error_message = f"Parameters are not found. Error: {e}\n"
            print(error_message)
            self.log(error_message)

        self.log(lines)
