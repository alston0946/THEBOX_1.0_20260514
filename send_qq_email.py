# -*- coding: utf-8 -*-
"""
把 output 目录下的 CSV 结果打包后通过 QQ 邮箱 SMTP 发送。

需要在 GitHub Secrets 中配置：
- QQ_MAIL_USER：发件 QQ 邮箱，例如 xxxxx@qq.com
- QQ_MAIL_AUTH_CODE：QQ 邮箱的 SMTP 授权码，不是 QQ 登录密码
- MAIL_TO：收件邮箱，可填一个或多个，多个用英文逗号分隔
"""

import glob
import os
import smtplib
import zipfile
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.qq.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
QQ_MAIL_USER = os.getenv("QQ_MAIL_USER", "").strip()
QQ_MAIL_AUTH_CODE = os.getenv("QQ_MAIL_AUTH_CODE", "").strip()
MAIL_TO = os.getenv("MAIL_TO", "").strip()
MAIL_SUBJECT_PREFIX = os.getenv("MAIL_SUBJECT_PREFIX", "箱体突破扫描结果").strip()


def now_cn_str() -> str:
    return (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def split_recipients(raw: str):
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]


def build_zip(csv_files):
    OUTPUT_DIR.mkdir(exist_ok=True)
    zip_path = OUTPUT_DIR / "box_breakout_results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in csv_files:
            p = Path(file_path)
            zf.write(p, arcname=p.name)
    return zip_path


def count_csv_rows(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            lines = sum(1 for _ in f)
        return max(0, lines - 1)
    except Exception:
        return -1


def main():
    if not QQ_MAIL_USER:
        raise ValueError("未检测到 QQ_MAIL_USER，请在 GitHub Secrets 中配置发件 QQ 邮箱。")
    if not QQ_MAIL_AUTH_CODE:
        raise ValueError("未检测到 QQ_MAIL_AUTH_CODE，请在 GitHub Secrets 中配置 QQ 邮箱 SMTP 授权码。")

    recipients = split_recipients(MAIL_TO)
    if not recipients:
        raise ValueError("未检测到 MAIL_TO，请在 GitHub Secrets 中配置收件邮箱。")

    csv_files = sorted(glob.glob(str(OUTPUT_DIR / "*.csv")))
    summary_lines = [
        f"运行时间（北京时间）：{now_cn_str()}",
        f"结果目录：{OUTPUT_DIR}",
        "",
    ]

    if csv_files:
        summary_lines.append("本次生成的 CSV 文件：")
        for file_path in csv_files:
            rows = count_csv_rows(file_path)
            row_text = f"{rows} 行" if rows >= 0 else "行数读取失败"
            summary_lines.append(f"- {Path(file_path).name}：{row_text}")
        zip_path = build_zip(csv_files)
    else:
        summary_lines.append("本次没有在 output 目录找到 CSV 文件，请检查 GitHub Actions 日志。")
        zip_path = None

    subject = f"{MAIL_SUBJECT_PREFIX} - {now_cn_str()}"

    msg = EmailMessage()
    msg["From"] = QQ_MAIL_USER
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content("\n".join(summary_lines))

    if zip_path and zip_path.exists():
        with open(zip_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="zip",
                filename=zip_path.name,
            )

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
        smtp.login(QQ_MAIL_USER, QQ_MAIL_AUTH_CODE)
        smtp.send_message(msg)

    print(f"邮件已发送到: {', '.join(recipients)}")


if __name__ == "__main__":
    main()
