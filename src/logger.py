import datetime
import os

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")


class Logger:
    def __init__(self) -> None:
        now = (datetime.datetime.now(JST)).strftime("%Y%m%d%H%M%S")
        self.file_path = f"output/log/log_{now}.txt"
        if not os.path.exists(self.file_path):
            with open(self.file_path, "a") as log:
                log.write(f"Log: {now}\n")

    def log(self, content: str) -> None:
        with open(self.file_path, "a") as log:
            log.write(f"{content}\n")
