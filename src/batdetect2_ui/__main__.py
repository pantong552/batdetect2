import argparse
import sys
import webbrowser
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="BatDetect2 Training Studio")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="綁定主機位址 (預設: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="監聽埠號 (預設: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="啟動時不自動開啟瀏覽器")
    parser.add_argument("--reload", action="store_true", help="開發模式熱重載")

    args = parser.parse_args()
    url = f"http://{args.host}:{args.port}"

    print(f"\n========================================================")
    print(f" [*] BatDetect2 Training Studio 正在啟動...")
    print(f" [>] 介面網址: {url}")
    print(f"========================================================\n")

    if not args.no_browser:
        # 延遲開啟瀏覽器
        import threading
        import time

        def open_browser():
            time.sleep(1.2)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        "batdetect2_ui.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
