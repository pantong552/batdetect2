import asyncio
import datetime
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket


ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class TrainingManager:
    """管理 BatDetect2 模型訓練程序、日誌捕捉與即時 WebSocket 廣播。"""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.process: Optional[asyncio.subprocess.Process] = None
        self.status: str = "idle"  # idle, training, completed, failed, stopped
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.current_epoch: int = 0
        self.total_epochs: int = 100
        self.current_step: int = 0
        self.total_steps: int = 0
        self.metrics_history: List[Dict[str, Any]] = []
        self.latest_metrics: Dict[str, Any] = {}
        self.logs_buffer: List[str] = []
        self.max_logs: int = 4000
        self.active_websockets: Set[WebSocket] = set()
        self._reader_task: Optional[asyncio.Task] = None
        self.current_config: Dict[str, Any] = {}

    def get_status_payload(self) -> Dict[str, Any]:
        elapsed = 0.0
        if self.start_time:
            if self.end_time:
                elapsed = self.end_time - self.start_time
            else:
                elapsed = time.time() - self.start_time

        return {
            "status": self.status,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "elapsed_seconds": round(elapsed, 1),
            "latest_metrics": self.latest_metrics,
            "metrics_history": self.metrics_history,
            "current_config": self.current_config,
            "is_running": self.status == "training",
        }

    async def connect_ws(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.add(websocket)
        # 發送當前完整狀態與最近日誌
        await websocket.send_json({
            "type": "init_state",
            "data": {
                **self.get_status_payload(),
                "logs": self.logs_buffer[-500:],  # 最近 500 行
            },
        })

    def disconnect_ws(self, websocket: WebSocket):
        self.active_websockets.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        dead_sockets = set()
        for ws in self.active_websockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.add(ws)
        for dead in dead_sockets:
            self.active_websockets.discard(dead)

    async def start_training(
        self,
        train_dataset: str,
        val_dataset: Optional[str] = None,
        targets_config: Optional[str] = None,
        model_path: Optional[str] = None,
        num_epochs: int = 100,
        train_workers: int = 0,
        val_workers: int = 0,
        seed: int = 42,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        custom_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.status == "training":
            return {"success": False, "message": "已有訓練任務進行中！"}

        self.status = "training"
        self.start_time = time.time()
        self.end_time = None
        self.current_epoch = 0
        self.total_epochs = num_epochs
        self.current_step = 0
        self.metrics_history = []
        self.latest_metrics = {}
        self.logs_buffer = []

        self.current_config = {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "targets_config": targets_config,
            "model_path": model_path,
            "num_epochs": num_epochs,
            "train_workers": train_workers,
            "val_workers": val_workers,
            "seed": seed,
            "experiment_name": experiment_name or "batdetect2_studio",
            "run_name": run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }

        # 建構執行指令
        # 使用 batdetect2 train CLI
        cmd = [
            sys.executable,
            "-m",
            "batdetect2",
            "train",
            train_dataset,
            "--num-epochs",
            str(num_epochs),
            "--train-workers",
            str(train_workers),
            "--val-workers",
            str(val_workers),
            "--seed",
            str(seed),
        ]

        if val_dataset:
            cmd.extend(["--val-dataset", val_dataset])
        if targets_config:
            cmd.extend(["--targets", targets_config])
        if model_path and model_path.strip():
            cmd.extend(["--model", model_path.strip()])
        if experiment_name:
            cmd.extend(["--experiment-name", experiment_name])
        if run_name:
            cmd.extend(["--run-name", run_name])
        if custom_args:
            cmd.extend(custom_args)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        self._append_log(f"[Studio] Start training task: {' '.join(cmd)}")
        await self.broadcast({
            "type": "training_started",
            "data": self.get_status_payload(),
        })

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.workspace_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self._reader_task = asyncio.create_task(self._process_stream())
            return {"success": True, "message": "訓練已成功啟動！"}
        except Exception as e:
            self.status = "failed"
            self.end_time = time.time()
            self._append_log(f"[Studio Error] Failed to start training: {str(e)}")
            await self.broadcast({
                "type": "training_failed",
                "data": {"error": str(e), **self.get_status_payload()},
            })
            return {"success": False, "message": f"啟動失敗: {str(e)}"}

    async def stop_training(self) -> Dict[str, Any]:
        if self.status != "training" or not self.process:
            return {"success": False, "message": "目前沒有正在進行的訓練！"}

        self._append_log("[Studio] Stop request received. Terminating process...")
        try:
            self.process.terminate()
            for _ in range(30):
                if self.process.returncode is not None:
                    break
                await asyncio.sleep(0.1)
            if self.process.returncode is None:
                self.process.kill()
        except Exception as e:
            self._append_log(f"[Studio Warning] Terminate process exception: {e}")

        self.status = "stopped"
        self.end_time = time.time()
        await self.broadcast({
            "type": "training_stopped",
            "data": self.get_status_payload(),
        })
        return {"success": True, "message": "訓練已成功停止。"}

    def _append_log(self, line: str):
        line = line.rstrip("\r\n")
        self.logs_buffer.append(line)
        if len(self.logs_buffer) > self.max_logs:
            self.logs_buffer.pop(0)

    async def _process_stream(self):
        """非同步讀取訓練標準輸出，提取 metrics 並格式化乾淨日誌。"""
        if not self.process or not self.process.stdout:
            return

        epoch_re = re.compile(r"Epoch\s+(\d+):", re.IGNORECASE)
        train_loss_re = re.compile(r"total_loss/train=([0-9\.\+\-eE]+)")
        val_loss_re = re.compile(r"total_loss/val=([0-9\.\+\-eE]+)")
        lr_re = re.compile(r"lr(?:-[a-zA-Z0-9]+)?=([0-9\.\+\-eE]+)")

        last_printed_epoch = -1

        while True:
            line_bytes = await self.process.stdout.readline()
            if not line_bytes:
                break

            try:
                line = line_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    line = line_bytes.decode("cp950")
                except UnicodeDecodeError:
                    line = line_bytes.decode("utf-8", errors="replace")

            line = line.rstrip("\r\n").strip()
            if not line:
                continue

            # 清理 ANSI 游標控制字符
            line = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line)
            line = re.sub(r"\[A|\[K|\[B", "", line)
            line = re.sub(r"[\ufffd]+", "#", line).strip()

            if not line:
                continue

            clean_line = ANSI_ESCAPE.sub("", line)

            # 解析指標
            metrics_updated = False
            parsed_epoch = None
            parsed_train_loss = None
            parsed_val_loss = None
            parsed_lr = None

            epoch_match = epoch_re.search(clean_line)
            if epoch_match:
                parsed_epoch = int(epoch_match.group(1))
                self.current_epoch = parsed_epoch

            train_loss_match = train_loss_re.search(clean_line)
            if train_loss_match:
                try:
                    parsed_train_loss = float(train_loss_match.group(1))
                    self.latest_metrics["train_loss"] = parsed_train_loss
                    metrics_updated = True
                except ValueError:
                    pass

            val_loss_match = val_loss_re.search(clean_line)
            if val_loss_match:
                try:
                    parsed_val_loss = float(val_loss_match.group(1))
                    self.latest_metrics["val_loss"] = parsed_val_loss
                    metrics_updated = True
                except ValueError:
                    pass

            lr_match = lr_re.search(clean_line)
            if lr_match:
                try:
                    parsed_lr = float(lr_match.group(1))
                    self.latest_metrics["learning_rate"] = parsed_lr
                    metrics_updated = True
                except ValueError:
                    pass

            # 記錄 Epoch 數值
            if parsed_epoch is not None and (parsed_train_loss is not None or parsed_val_loss is not None):
                if self.metrics_history and self.metrics_history[-1].get("epoch") == parsed_epoch:
                    if parsed_train_loss is not None:
                        self.metrics_history[-1]["train_loss"] = parsed_train_loss
                    if parsed_val_loss is not None:
                        self.metrics_history[-1]["val_loss"] = parsed_val_loss
                    if parsed_lr is not None:
                        self.metrics_history[-1]["learning_rate"] = parsed_lr
                else:
                    self.metrics_history.append({
                        "epoch": parsed_epoch,
                        "train_loss": parsed_train_loss,
                        "val_loss": parsed_val_loss,
                        "learning_rate": parsed_lr,
                        "timestamp": round(time.time() - (self.start_time or time.time()), 2),
                    })

            # 過濾純 Validation / Sanity Checking 暫態中間步雜訊
            if re.search(r"^(Validation|Sanity Checking)(?:\s+DataLoader\s+\d+)?:", clean_line, re.IGNORECASE):
                if metrics_updated:
                    await self.broadcast({
                        "type": "log_line",
                        "data": {
                            "line": None,
                            "metrics_updated": True,
                            "status": self.get_status_payload(),
                        },
                    })
                continue

            # 格式化 Epoch 輸出行，清楚呈現 Epoch 序號與 Loss，並防止重複行
            display_line = line
            if parsed_epoch is not None and (parsed_train_loss is not None or parsed_val_loss is not None):
                # 只有在 val_loss 有數值，或是最後一步時輸出，避免同一個 epoch 輸出兩次
                if parsed_val_loss is not None or parsed_epoch != last_printed_epoch:
                    t_loss_str = f"{parsed_train_loss:.4f}" if parsed_train_loss is not None else "--"
                    v_loss_str = f"{parsed_val_loss:.4f}" if parsed_val_loss is not None else "--"
                    display_line = f">> [Epoch {parsed_epoch + 1:03d}/{self.total_epochs:03d}] Train Loss: {t_loss_str} | Val Loss: {v_loss_str}"
                    last_printed_epoch = parsed_epoch
                else:
                    continue

            self._append_log(display_line)

            await self.broadcast({
                "type": "log_line",
                "data": {
                    "line": display_line,
                    "clean_line": ANSI_ESCAPE.sub("", display_line),
                    "metrics_updated": metrics_updated,
                    "status": self.get_status_payload(),
                },
            })

        return_code = await self.process.wait()
        self.end_time = time.time()
        if return_code == 0:
            self.status = "completed"
            self._append_log("[Studio] Training successfully completed!")
            await self.broadcast({
                "type": "training_completed",
                "data": self.get_status_payload(),
            })
        elif self.status != "stopped":
            self.status = "failed"
            self._append_log(f"[Studio] Training exited with code {return_code}")
            await self.broadcast({
                "type": "training_failed",
                "data": {"exit_code": return_code, **self.get_status_payload()},
            })
