import base64
import glob
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import fastapi
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from batdetect2_ui.training_runner import TrainingManager


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOADS_DIR = WORKSPACE_ROOT / "outputs" / "ui_uploads"
CHECKPOINTS_DIR = WORKSPACE_ROOT / "outputs" / "checkpoints"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="BatDetect2 Training Studio",
    description="專業現代化 BatDetect2 模型訓練與深度學習音訊監控工作台",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_manager = TrainingManager(workspace_root=WORKSPACE_ROOT)


@app.get("/api/system-info")
async def get_system_info():
    """取得系統硬體規格（GPU / CUDA / CPU / RAM）。"""
    cuda_available = torch.cuda.is_available()
    gpu_info = []
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total_gb = round(props.total_memory / (1024 ** 3), 2)
            mem_alloc_gb = round(torch.cuda.memory_allocated(i) / (1024 ** 3), 2)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": mem_total_gb,
                "allocated_memory_gb": mem_alloc_gb,
                "compute_capability": f"{props.major}.{props.minor}",
            })

    ram = psutil.virtual_memory()
    return {
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpus": gpu_info,
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_total_gb": round(ram.total / (1024 ** 3), 2),
        "ram_used_gb": round(ram.used / (1024 ** 3), 2),
        "ram_percent": ram.percent,
        "workspace_root": str(WORKSPACE_ROOT),
    }


@app.get("/api/configs/presets")
async def get_config_presets():
    """取得專案內預設的資料集與 targets 設定檔路徑與內容預覽。"""
    presets = {
        "datasets": [],
        "targets": [],
    }

    # 搜尋 example_data 及專案下的 yaml 檔案
    for p in WORKSPACE_ROOT.glob("**/*.yaml"):
        if ".venv" in p.parts or ".git" in p.parts:
            continue
        rel_path = str(p.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
        try:
            content = p.read_text(encoding="utf-8")
            item = {
                "name": p.name,
                "path": rel_path,
                "preview": content[:600],
            }
            if "target" in p.name.lower():
                presets["targets"].append(item)
            else:
                presets["datasets"].append(item)
        except Exception:
            pass

    return presets


@app.post("/api/train/start")
async def start_training(payload: Dict[str, Any]):
    """啟動模型訓練。"""
    train_dataset = payload.get("train_dataset", "example_data/dataset.yaml")
    val_dataset = payload.get("val_dataset", "example_data/dataset.yaml")
    targets_config = payload.get("targets_config", "example_data/targets.yaml")
    model_path = payload.get("model_path")
    num_epochs = int(payload.get("num_epochs", 100))
    train_workers = int(payload.get("train_workers", 0))
    val_workers = int(payload.get("val_workers", 0))
    seed = int(payload.get("seed", 42))
    experiment_name = payload.get("experiment_name", "batdetect2_studio")
    run_name = payload.get("run_name")

    res = await training_manager.start_training(
        train_dataset=train_dataset,
        val_dataset=val_dataset if val_dataset else None,
        targets_config=targets_config if targets_config else None,
        model_path=model_path if model_path else None,
        num_epochs=num_epochs,
        train_workers=train_workers,
        val_workers=val_workers,
        seed=seed,
        experiment_name=experiment_name,
        run_name=run_name,
    )
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.post("/api/train/stop")
async def stop_training():
    """終止訓練程序。"""
    res = await training_manager.stop_training()
    return res


@app.get("/api/train/status")
async def get_train_status():
    """取得當前訓練狀態與指標。"""
    return training_manager.get_status_payload()


@app.get("/api/checkpoints")
async def list_checkpoints():
    """取得 outputs/checkpoints/ 下的所有 Checkpoints。"""
    checkpoints = []
    if CHECKPOINTS_DIR.exists():
        for f in CHECKPOINTS_DIR.glob("**/*.ckpt"):
            stat = f.stat()
            rel_path = str(f.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
            checkpoints.append({
                "filename": f.name,
                "relative_path": rel_path,
                "absolute_path": str(f),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created_time": stat.st_ctime,
                "is_best": "best" in f.name.lower() or "last" not in f.name.lower(),
            })

    # 排序：最新產生的排前面
    checkpoints.sort(key=lambda x: x["created_time"], reverse=True)
    return checkpoints


@app.delete("/api/checkpoints")
async def delete_checkpoint(path: str):
    """刪除指定的 Checkpoint 檔案。"""
    target_path = (WORKSPACE_ROOT / path).resolve()
    # 安全檢查：必須在 CHECKPOINTS_DIR 內且為 .ckpt 檔案
    if not str(target_path).startswith(str(CHECKPOINTS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="禁止刪除 checkpoints 目錄外的檔案！")
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="該 Checkpoint 檔案不存在！")

    try:
        os.remove(target_path)
        return {"success": True, "message": "檔案已成功刪除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除失敗: {str(e)}")


@app.get("/api/examples")
async def list_examples():
    """取得 example_data/audio/ 中的範例音訊。"""
    audio_dir = WORKSPACE_ROOT / "example_data" / "audio"
    examples = []
    if audio_dir.exists():
        for f in audio_dir.glob("*.wav"):
            examples.append({
                "filename": f.name,
                "path": str(f.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            })
    return examples


@app.get("/api/audio")
async def stream_audio(path: str):
    """提供音訊檔案串流。"""
    file_path = (WORKSPACE_ROOT / path).resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="音訊檔案不存在")
    return FileResponse(str(file_path), media_type="audio/wav")


@app.post("/api/predict")
async def run_prediction(
    file: Optional[UploadFile] = File(None),
    preset_path: Optional[str] = Form(None),
    detection_threshold: float = Form(0.3),
    max_duration: float = Form(3.0),
    time_expansion: float = Form(1.0),
):
    """執行模型推論並回傳時頻圖影像與標註清單。"""
    import batdetect2.api as api
    import batdetect2.plotting.legacy.plot as plot

    target_audio_path = None
    if file and file.filename:
        save_path = UPLOADS_DIR / file.filename
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        target_audio_path = str(save_path)
    elif preset_path:
        target_audio_path = str(WORKSPACE_ROOT / preset_path)
    else:
        raise HTTPException(status_code=400, detail="請提供音訊檔案或選擇範例檔案！")

    try:
        # 設定推論 config
        run_config = api.get_config(
            detection_threshold=detection_threshold,
            max_duration=max_duration,
            time_expansion=time_expansion,
        )

        results = api.process_file(target_audio_path, config=run_config)
        detections = results.get("pred_dict", {}).get("annotation", [])

        # 產生頻譜圖
        audio = api.load_audio(
            target_audio_path,
            max_duration=run_config["max_duration"],
            time_exp_fact=run_config["time_expansion"],
            target_samp_rate=run_config["target_samp_rate"],
        )
        spec = api.generate_spectrogram(audio, config=run_config)

        plt.close("all")
        fig = plt.figure(1, figsize=(14, 4.2), dpi=120, facecolor="#181818")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#181818")
        
        # 繪製頻譜與標註
        plot.spectrogram_with_detections(spec, detections, ax=ax)
        
        # 美化刻度樣式 (VS Code Dark)
        ax.tick_params(colors="#858585", labelsize=8.5)
        ax.xaxis.label.set_color("#858585")
        ax.yaxis.label.set_color("#858585")
        for spine in ax.spines.values():
            spine.set_color("#2d2d2d")
            
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # 格式化 detections
        formatted_detections = []
        for i, d in enumerate(detections):
            formatted_detections.append({
                "id": i + 1,
                "species": d.get("class", "Bat"),
                "start_time": round(float(d.get("start_time", 0.0)), 4),
                "end_time": round(float(d.get("end_time", 0.0)), 4),
                "duration_ms": round((float(d.get("end_time", 0.0)) - float(d.get("start_time", 0.0))) * 1000, 1),
                "low_freq": round(float(d.get("low_freq", 0.0)) / 1000, 1),  # kHz
                "high_freq": round(float(d.get("high_freq", 0.0)) / 1000, 1),  # kHz
                "confidence": round(float(d.get("class_prob", d.get("det_prob", 0.0))), 3),
                "event": d.get("event", "Echolocation"),
            })

        return {
            "success": True,
            "spectrogram_image": f"data:image/png;base64,{img_b64}",
            "detections": formatted_detections,
            "total_detections": len(formatted_detections),
            "audio_duration": round(len(audio) / run_config["target_samp_rate"], 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推論失敗: {str(e)}")


@app.websocket("/ws/train")
async def websocket_train_endpoint(websocket: WebSocket):
    """即時串流訓練日誌與指標的 WebSocket 端點。"""
    await training_manager.connect_ws(websocket)
    try:
        while True:
            # 接收客戶端指令（例如即時 ping 或指令）
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        training_manager.disconnect_ws(websocket)
    except Exception:
        training_manager.disconnect_ws(websocket)


# 掛載前端靜態檔案
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
