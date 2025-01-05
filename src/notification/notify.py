import os
import smtplib
from email.mime.text import MIMEText

from dotenv import load_dotenv


def send_email(msg: str):
    load_dotenv()
    from_addr = str(os.environ["FROM_ADDR"])
    to_addr = str(os.environ["TO_ADDR"])
    password = str(os.environ["GOOGLE_APP_PASS"])
    smtp_server = str(os.environ["SMTP_SERVER"])
    smtp_port = int(os.environ["SMTP_PORT"])

    try:
        connection = smtplib.SMTP(smtp_server, smtp_port)
        connection.set_debuglevel(True)
        connection.starttls()
        connection.login(from_addr, password)
        connection.sendmail(from_addr, to_addr, msg)
    except Exception as e:
        print(e)
