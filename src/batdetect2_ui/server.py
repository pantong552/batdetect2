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
    trainable = payload.get("trainable", "heads")
    audio_config = payload.get("audio_config")
    preprocess_config = payload.get("preprocess_config")
    training_config = payload.get("training_config")

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
        trainable=trainable,
        audio_config=audio_config,
        preprocess_config=preprocess_config,
        training_config=training_config,
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


@app.get("/api/experiments")
async def list_experiments():
    """取得 outputs/logs/ 下的所有實驗版本（version_0, version_1...）。"""
    logs_dir = WORKSPACE_ROOT / "outputs" / "logs"
    experiments = []
    if logs_dir.exists():
        for item in logs_dir.iterdir():
            if item.is_dir() and item.name.startswith("version_"):
                stat = item.stat()
                metrics_csv = item / "metrics.csv"
                hparams_yaml = item / "hparams.yaml"
                experiments.append({
                    "id": item.name,
                    "name": f"Training Run ({item.name})",
                    "path": str(item.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                    "created_time": stat.st_ctime,
                    "has_metrics": metrics_csv.exists(),
                    "has_hparams": hparams_yaml.exists(),
                })

    experiments.sort(key=lambda x: x["created_time"], reverse=True)
    return experiments


@app.get("/api/experiments/{exp_id}")
async def get_experiment_details(exp_id: str):
    """讀取指定實驗的 metrics.csv 曲線數據與 hparams.yaml 超參數設定。"""
    logs_dir = WORKSPACE_ROOT / "outputs" / "logs"
    if exp_id == "latest":
        # 自動搜尋最新修改的 version_X 資料夾
        versions = [p for p in logs_dir.iterdir() if p.is_dir() and p.name.startswith("version_")]
        if not versions:
            raise HTTPException(status_code=404, detail="尚無任何實驗日誌！")
        exp_dir = max(versions, key=lambda p: p.stat().st_mtime)
        exp_id = exp_dir.name
    else:
        exp_dir = logs_dir / exp_id
        if not exp_dir.exists():
            raise HTTPException(status_code=404, detail="找不到該實驗日誌！")

    metrics_data = []
    columns = []
    aggregated_metrics = []
    metrics_csv = exp_dir / "metrics.csv"
    if metrics_csv.exists():
        import csv
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            epoch_map = {}
            for row in reader:
                parsed_row = {}
                for k, v in row.items():
                    if v is not None and v != "":
                        try:
                            parsed_row[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                        except ValueError:
                            parsed_row[k] = v
                    else:
                        parsed_row[k] = None
                metrics_data.append(parsed_row)

                # 依據 Epoch 進行智能合併（PyTorch Lightning 分別寫入 train 與 val）
                epoch_val = parsed_row.get("epoch")
                if epoch_val is not None:
                    epoch_int = int(epoch_val)
                    if epoch_int not in epoch_map:
                        epoch_map[epoch_int] = {"epoch": epoch_int}
                    # 將非 None 的指標更新進去
                    for k, val in parsed_row.items():
                        if val is not None:
                            epoch_map[epoch_int][k] = val

            # 依 epoch 排序
            aggregated_metrics = [epoch_map[ep] for ep in sorted(epoch_map.keys())]

    hparams_data = {}
    hparams_yaml = exp_dir / "hparams.yaml"
    if hparams_yaml.exists():
        import yaml
        try:
            with open(hparams_yaml, "r", encoding="utf-8") as f:
                hparams_data = yaml.safe_load(f) or {}
        except Exception:
            pass

    return {
        "id": exp_id,
        "columns": columns,
        "metrics": aggregated_metrics if aggregated_metrics else metrics_data,
        "raw_metrics": metrics_data,
        "hparams": hparams_data,
        "total_rows": len(aggregated_metrics if aggregated_metrics else metrics_data),
    }


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
    model_path: Optional[str] = Form(None),
    detection_threshold: float = Form(0.3),
    max_duration: float = Form(3.0),
    time_expansion: float = Form(1.0),
):
    """執行模型推論並回傳時頻圖影像與標註清單。"""
    import batdetect2.api as api
    import batdetect2.plotting.legacy.plot as plot

    target_audio_path = None
    audio_stream_rel_path = None

    if file and file.filename:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = UPLOADS_DIR / file.filename
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        target_audio_path = str(save_path.resolve())
        audio_stream_rel_path = str(save_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    elif preset_path:
        preset_file = (WORKSPACE_ROOT / preset_path).resolve()
        if not preset_file.exists():
            raise HTTPException(status_code=404, detail=f"找不到指定的範例音訊檔案: {preset_path}")
        target_audio_path = str(preset_file)
        audio_stream_rel_path = preset_path
    else:
        raise HTTPException(status_code=400, detail="請提供音訊檔案或選擇範例檔案！")

    try:
        model_obj = api.MODEL
        run_config = api.get_config(
            detection_threshold=detection_threshold,
            max_duration=max_duration,
            time_expansion=time_expansion,
        )

        if model_path and model_path.strip():
            ckpt_str = model_path.strip()
            if ckpt_str.endswith(".pth.tar"):
                full_model_path = str((WORKSPACE_ROOT / ckpt_str).resolve()) if not Path(ckpt_str).is_absolute() else ckpt_str
                loaded_model, params = api.load_model(full_model_path, device=api.DEVICE)
                model_obj = loaded_model
                run_config = {**run_config, **params}

        # 若 max_duration <= 0 則載入整首音訊不限長度
        load_max_dur = max_duration if (max_duration and max_duration > 0) else None

        # 讀取音訊檔案真實採樣率 (Sampling Rate)
        import soundfile as sf
        audio_info = sf.info(target_audio_path)
        actual_sr = audio_info.samplerate

        detections = []
        spec = None

        if model_path and model_path.strip() and model_path.strip().endswith(".ckpt"):
            # 使用 BatDetect2API V2 原生推論管線
            from batdetect2.api_v2 import BatDetect2API
            full_ckpt_path = str((WORKSPACE_ROOT / model_path.strip()).resolve()) if not Path(model_path.strip()).is_absolute() else model_path.strip()
            v2_api = BatDetect2API.from_checkpoint(full_ckpt_path)
            
            # 執行推論
            clip_res = v2_api.process_file(target_audio_path, detection_threshold=detection_threshold)
            
            # 依據 wav 檔案自身真實 sampling rate 或模型需求動態載入音訊
            target_sr = actual_sr
            run_config["target_samp_rate"] = target_sr
            # 若採樣率高於 256k (例如 384k/500k)，動態更新時頻圖頻率上限為 Nyquist (sr // 2) 或保留配置
            if target_sr > 256000:
                run_config["max_freq"] = max(run_config.get("max_freq", 120000), target_sr // 2)

            audio = api.load_audio(
                target_audio_path,
                max_duration=load_max_dur,
                time_exp_fact=time_expansion,
                target_samp_rate=target_sr,
            )
            spec = api.generate_spectrogram(audio, samp_rate=target_sr, config=run_config)

            # 將 Soundevent / V2 Detections 轉換為統一格式
            for det in clip_res.detections:
                geom = det.geometry
                top_class = v2_api.get_top_class_name(det)
                
                if hasattr(geom, "coordinates"):
                    coords = geom.coordinates
                    start_t, low_f, end_t, high_f = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                elif hasattr(geom, "start_time"):
                    start_t, low_f = float(geom.start_time), float(geom.low_freq)
                    end_t, high_f = float(geom.end_time), float(geom.high_freq)
                else:
                    continue

                detections.append({
                    "start_time": round(start_t, 4),
                    "end_time": round(end_t, 4),
                    "low_freq": round(low_f, 1),
                    "high_freq": round(high_f, 1),
                    "class": top_class,
                    "class_prob": round(float(np.max(det.class_scores)), 3),
                    "det_prob": round(float(det.detection_score), 3),
                    "event": "Echolocation",
                })
        else:
            # 使用 Legacy API 推論
            target_sr = actual_sr
            run_config["target_samp_rate"] = target_sr
            if target_sr > 256000:
                run_config["max_freq"] = max(run_config.get("max_freq", 120000), target_sr // 2)

            results = api.process_file(target_audio_path, model=model_obj, config=run_config)
            detections = results.get("pred_dict", {}).get("annotation", [])
            audio = api.load_audio(
                target_audio_path,
                max_duration=load_max_dur,
                time_exp_fact=run_config["time_expansion"],
                target_samp_rate=target_sr,
            )
            spec = api.generate_spectrogram(audio, samp_rate=target_sr, config=run_config)

        audio_dur = len(audio) / run_config["target_samp_rate"]
        
        # 動態計算圖像寬度 (每秒約 5 inches，最小 14 inches)，高度設為 7.5 inches 以適應 600px 高解析度
        calc_width = max(14.0, min(audio_dur * 5.0, 100.0))
        
        plt.close("all")
        fig = plt.figure(1, figsize=(calc_width, 7.5), dpi=120, facecolor="#181818")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#181818")
        
        # 繪製頻譜與在音訊長度範圍內的所有標註 (嚴格過濾門檻與頂部/底部邊緣偽影)
        plot_dets = []
        max_valid_freq = run_config.get("max_freq", 128000) * 0.95
        min_valid_freq = run_config.get("min_freq", 10000) * 1.05
        
        for d in detections:
            if d.get("start_time", 0) > audio_dur:
                continue
            det_score = float(d.get("det_prob", d.get("class_prob", 0.0)))
            if detection_threshold is not None and det_score < detection_threshold:
                continue
            # 排除壓在頂部或底部的 STFT 頻率裁切偽影
            if float(d.get("high_freq", 0.0)) > max_valid_freq and float(d.get("low_freq", 0.0)) > max_valid_freq * 0.85:
                continue
            plot_dets.append(d)

        plot.spectrogram_with_detections(spec, plot_dets, config=run_config, ax=ax)
        
        # 鎖定 X/Y 軸範圍與時頻圖完全一致
        ax.set_xlim(0, audio_dur)
        ax.set_ylim(run_config["min_freq"], run_config["max_freq"])

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

        # 格式化 detections 並嚴格依據 detection_threshold 與邊界偽影過濾
        formatted_detections = []
        for i, d in enumerate(detections):
            det_score = float(d.get("det_prob", d.get("class_prob", 0.0)))
            cls_score = float(d.get("class_prob", det_score))
            
            # 若後端推論時未過濾，在此做嚴格門檻過濾
            if detection_threshold is not None and det_score < detection_threshold:
                continue

            # 排除壓在頂部的 STFT 頻率裁切偽影
            if float(d.get("high_freq", 0.0)) > max_valid_freq and float(d.get("low_freq", 0.0)) > max_valid_freq * 0.85:
                continue

            formatted_detections.append({
                "id": len(formatted_detections) + 1,
                "species": d.get("class", "Bat"),
                "start_time": round(float(d.get("start_time", 0.0)), 4),
                "end_time": round(float(d.get("end_time", 0.0)), 4),
                "duration_ms": round((float(d.get("end_time", 0.0)) - float(d.get("start_time", 0.0))) * 1000, 1),
                "low_freq": round(float(d.get("low_freq", 0.0)) / 1000, 1),  # kHz
                "high_freq": round(float(d.get("high_freq", 0.0)) / 1000, 1),  # kHz
                "confidence": round(det_score, 3),  # 叫聲偵測置信度 (例如 85.6%)
                "class_prob": round(cls_score, 3),  # 物種分類置信度
                "event": d.get("event", "Echolocation"),
            })

        return {
            "success": True,
            "spectrogram_image": f"data:image/png;base64,{img_b64}",
            "detections": formatted_detections,
            "total_detections": len(formatted_detections),
            "audio_duration": round(len(audio) / run_config["target_samp_rate"], 2),
            "audio_stream_path": audio_stream_rel_path,
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
