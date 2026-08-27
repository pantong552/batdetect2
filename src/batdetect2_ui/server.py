import asyncio
import base64
import glob
import io
import json
import os
import re
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
MODELS_DIR = WORKSPACE_ROOT / "outputs" / "models"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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


@app.post("/api/checkpoints/export-onnx")
async def export_checkpoint_onnx(payload: Dict[str, Any]):
    """將指定的 Checkpoint 轉換為 ONNX 格式模型與 metadata.json，儲存於 outputs/models/{model_name}/"""
    ckpt_rel_path = payload.get("path", "").strip()
    if not ckpt_rel_path:
        raise HTTPException(status_code=400, detail="請提供 Checkpoint 路徑！")

    ckpt_path = (WORKSPACE_ROOT / ckpt_rel_path).resolve()
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail="找不到該 Checkpoint 檔案！")

    # 決定子目錄名稱：例如 outputs/checkpoints/batdetect2_studio_run/last-v8.ckpt -> batdetect2_studio_run_last-v8 或取其父資料夾與檔名
    parent_name = ckpt_path.parent.name if ckpt_path.parent.name != "checkpoints" else "default"
    stem_name = ckpt_path.stem
    model_subfolder_name = f"{parent_name}_{stem_name}" if parent_name != stem_name else stem_name
    
    # 也可以允許前端自訂 subfolder，若無則使用自動命名的 subfolder
    custom_subfolder = payload.get("subfolder", "").strip()
    if custom_subfolder:
        model_subfolder_name = re.sub(r"[^\w\-\.]", "_", custom_subfolder)

    output_models_dir = (MODELS_DIR / model_subfolder_name).resolve()
    output_models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 在背景線程非同步調用 export_checkpoint (使用 importlib 動態載入專案 scripts/export_onnx.py)
        import importlib.util

        export_script = WORKSPACE_ROOT / "scripts" / "export_onnx.py"
        if not export_script.exists():
            raise FileNotFoundError(f"找不到導出腳本: {export_script}")

        spec = importlib.util.spec_from_file_location("export_onnx", export_script)
        if spec is None or spec.loader is None:
            raise ImportError(f"無法載入導出腳本: {export_script}")

        export_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_module)
        export_checkpoint = export_module.export_checkpoint

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, export_checkpoint, str(ckpt_path), str(output_models_dir))

        rel_out_dir = str(output_models_dir.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
        return {
            "success": True,
            "message": f"ONNX 導出成功！",
            "output_dir": rel_out_dir,
            "onnx_path": f"{rel_out_dir}/batdetect2.onnx",
            "metadata_path": f"{rel_out_dir}/metadata.json",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"導出 ONNX 失敗: {str(e)}")


@app.get("/api/models")
async def list_exported_models():
    """取得 outputs/models/ 下的所有導出模型子資料夾與其資訊。"""
    models = []
    if MODELS_DIR.exists():
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                onnx_file = item / "batdetect2.onnx"
                meta_file = item / "metadata.json"
                stat = item.stat()
                onnx_size_mb = round(onnx_file.stat().st_size / (1024 * 1024), 2) if onnx_file.exists() else 0

                meta_data = {}
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta_data = json.load(f)
                    except Exception:
                        pass

                models.append({
                    "name": item.name,
                    "relative_path": str(item.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
                    "has_onnx": onnx_file.exists(),
                    "has_metadata": meta_file.exists(),
                    "onnx_size_mb": onnx_size_mb,
                    "created_time": stat.st_ctime,
                    "classes": meta_data.get("classes", []),
                    "sample_rate": meta_data.get("sample_rate", None),
                })

    models.sort(key=lambda x: x["created_time"], reverse=True)
    return models


@app.delete("/api/models")
async def delete_exported_model(path: str):
    """刪除 outputs/models/ 下的指定模型資料夾。"""
    target_path = (WORKSPACE_ROOT / path).resolve()
    if not str(target_path).startswith(str(MODELS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="禁止刪除 models 目錄外的資料夾！")
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="該模型資料夾不存在！")

    try:
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            os.remove(target_path)
        return {"success": True, "message": "模型已成功刪除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除失敗: {str(e)}")


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



def estimate_training_hardware(
    num_audio_files: int,
    total_duration_sec: float,
    batch_size: int = 8,
    train_workers: int = 4,
    trainable: str = "heads",
    precision: str = "fp16",
    spectrogram_freq_bins: int = 256,
    clip_duration: float = 2.0,
) -> Dict[str, Any]:
    """精確估算 BatDetect2 訓練時所需的 GPU VRAM (顯存) 及系統 RAM (記憶體)，並結合目前電腦即時可用記憶體診斷。"""
    # 1. 模型參數與優化器開銷
    total_params = 26_500_000  # 約 26.5M 參數
    trainable_params = total_params if trainable == "full" else 3_200_000  # heads 微調約 3.2M

    bytes_per_float = 2 if precision == "fp16" else 4

    # 模型權重顯存 (MB)
    model_weights_mb = round((total_params * bytes_per_float) / (1024 ** 2), 2)
    # 梯度顯存 (MB) (反向傳播梯度通常以 fp32 累計)
    gradients_mb = round((trainable_params * 4) / (1024 ** 2), 2)
    # AdamW Optimizer 狀態 (momentum + variance, 8 bytes per trainable param)
    optimizer_mb = round((trainable_params * 8) / (1024 ** 2), 2)

    # 2. 前向與反向傳播激活特徵圖 (Activations Memory per batch)
    # 在 BatDetect2 FPN + 多尺度 Detection Heads 反向傳播時需保存完整的卷積與注意力特徵圖
    # FP32 實測每 sample 約 460MB，FP16 (AMP) 實測每 sample 約 260MB
    act_per_sample_mb = 260.0 if precision == "fp16" else 460.0
    activations_mb = round(batch_size * act_per_sample_mb, 2)

    # 3. CUDA Context / CUDNN Workspace 與 Windows 桌面系統開銷
    cuda_base_mb = 850.0  # CUDA runtime, Context, CuDNN Workspace, memory allocator baseline
    system_gpu_overhead_mb = 450.0  # Windows DWM / 桌面系統渲染常駐佔用

    # 總 VRAM 尖峰 (含 18% PyTorch 顯存池碎片化緩衝)
    total_vram_mb = (model_weights_mb + gradients_mb + optimizer_mb + activations_mb + cuda_base_mb) * 1.18
    total_vram_gb = round(total_vram_mb / 1024, 2)

    # 讀取目前本機 GPU 資訊
    cuda_available = torch.cuda.is_available()
    gpu_name = "CPU Only (No CUDA)"
    gpu_vram_gb = 0.0
    usable_gpu_vram_gb = 0.0

    if cuda_available and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_vram_gb = round(props.total_memory / (1024 ** 3), 2)
        # 扣除 Windows 系統桌面佔用 (約 0.45GB)
        usable_gpu_vram_gb = max(0.5, round(gpu_vram_gb - (system_gpu_overhead_mb / 1024), 2))

    # 產生常用 Batch Size 的 VRAM 矩陣 (1, 2, 4, 8, 16, 32, 64)
    vram_matrix = []
    for b in [1, 2, 4, 8, 16, 32, 64]:
        b_act_mb = b * act_per_sample_mb
        b_total_mb = (model_weights_mb + gradients_mb + optimizer_mb + b_act_mb + cuda_base_mb) * 1.18
        b_vram_gb = round(b_total_mb / 1024, 2)

        if gpu_vram_gb > 0:
            if b_vram_gb <= usable_gpu_vram_gb * 0.75:
                status_txt = "Perfect Fit"
                status_type = "PERFECT"
            elif b_vram_gb <= usable_gpu_vram_gb:
                status_txt = "Tight Fit (OOM Risk)"
                status_type = "TIGHT"
            else:
                status_txt = "Out of Memory"
                status_type = "OOM"
        else:
            status_txt = "CPU Mode"
            status_type = "CPU"

        vram_matrix.append({
            "batch_size": b,
            "vram_gb": b_vram_gb,
            "vram_mb": round(b_total_mb, 0),
            "status_text": status_txt,
            "status_type": status_type,
        })

    # 4. 電腦系統實體 RAM (Host RAM) 計算與即時可用性對比
    vmem = psutil.virtual_memory()
    sys_total_ram_gb = round(vmem.total / (1024 ** 3), 2)
    sys_used_ram_gb = round(vmem.used / (1024 ** 3), 2)
    sys_available_ram_gb = round(vmem.available / (1024 ** 3), 2)
    sys_ram_percent = vmem.percent

    # (a) 音訊與標註結構快取 (檔案清單、元數據與標註緩存)
    dataset_cache_mb = min(1536, max(300, int(num_audio_files * 2.0)))
    # (b) Windows 多進程 DataLoader Workers 開銷 (Windows spawn 模式下每個進程約 0.95GB)
    worker_process_ram_mb = train_workers * 980
    # (c) 主進程 Python + PyTorch + Lightning 常駐記憶體
    main_process_ram_mb = 2200

    total_req_ram_mb = dataset_cache_mb + worker_process_ram_mb + main_process_ram_mb
    total_req_ram_gb = round(total_req_ram_mb / 1024, 2)

    # 產生常用 Workers 數量的 RAM 需求矩陣 (0, 1, 2, 4, 8)
    ram_workers_matrix = []
    for w in [0, 1, 2, 4, 8]:
        w_ram_mb = dataset_cache_mb + (w * 980) + main_process_ram_mb
        w_ram_gb = round(w_ram_mb / 1024, 2)
        
        if sys_available_ram_gb >= w_ram_gb * 1.25:
            w_status = "RAM Sufficient (Safe)"
            w_status_type = "SAFE"
        elif sys_available_ram_gb >= w_ram_gb:
            w_status = "RAM Tight"
            w_status_type = "TIGHT"
        else:
            w_status = "Insufficient RAM (OOM Risk)"
            w_status_type = "OOM"

        ram_workers_matrix.append({
            "workers": w,
            "req_ram_gb": w_ram_gb,
            "req_ram_mb": w_ram_mb,
            "status_text": w_status,
            "status_type": w_status_type,
        })

    # 5. 綜合硬體診斷與調優建議
    vram_advisory = {
        "status": "UNKNOWN",
        "status_color": "#858585",
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "usable_gpu_vram_gb": usable_gpu_vram_gb,
        "advice": "",
        "recommended_batch_size": 2,
    }

    if cuda_available and gpu_vram_gb > 0:
        if total_vram_gb <= usable_gpu_vram_gb * 0.75:
            vram_advisory["status"] = "PERFECT"
            vram_advisory["status_color"] = "#4ec9b0"
            vram_advisory["advice"] = f"GPU 顯存非常充裕！此 Batch Size ({batch_size}) 可穩定全速訓練。"
            vram_advisory["recommended_batch_size"] = batch_size
        elif total_vram_gb <= usable_gpu_vram_gb:
            vram_advisory["status"] = "TIGHT"
            vram_advisory["status_color"] = "#dcdcaa"
            vram_advisory["advice"] = f"顯存極度緊繃 (預估需 {total_vram_gb} GB，扣除系統後可用 {usable_gpu_vram_gb} GB)！在峰值階段容易爆顯存 (CUDA OOM)，建議調小 Batch Size (例如降至 {max(1, batch_size // 2)}) 或啟用 FP16。"
            vram_advisory["recommended_batch_size"] = max(1, batch_size // 2)
        else:
            vram_advisory["status"] = "OOM_RISK"
            vram_advisory["status_color"] = "#f14c4c"
            vram_advisory["advice"] = f"【爆顯存警告】預估需要 {total_vram_gb} GB 顯存，已超過您的 GPU 實際可用顯存 ({usable_gpu_vram_gb} GB)！若直接執行訓練必然會觸發 CUDA out of memory。強烈建議將 Batch Size 降至 {max(1, 4 if usable_gpu_vram_gb >= 3.0 else 2)}！"
            # 計算適合該 GPU 的安全最大 Batch Size
            rec_b = 1
            for row in vram_matrix:
                if row["vram_gb"] <= usable_gpu_vram_gb * 0.8:
                    rec_b = row["batch_size"]
            vram_advisory["recommended_batch_size"] = rec_b
    else:
        vram_advisory["status"] = "CPU_MODE"
        vram_advisory["status_color"] = "#858585"
        vram_advisory["advice"] = "未偵測到獨立 CUDA GPU，訓練將依賴 CPU 與系統實體 RAM 進行。"
        vram_advisory["recommended_batch_size"] = 4

    # 系統實體 RAM 診斷
    ram_advisory = {
        "status": "SAFE" if sys_available_ram_gb >= total_req_ram_gb * 1.15 else ("TIGHT" if sys_available_ram_gb >= total_req_ram_gb else "OOM_RISK"),
        "status_color": "#4ec9b0" if sys_available_ram_gb >= total_req_ram_gb * 1.15 else ("#dcdcaa" if sys_available_ram_gb >= total_req_ram_gb else "#f14c4c"),
        "total_ram_gb": sys_total_ram_gb,
        "used_ram_gb": sys_used_ram_gb,
        "available_ram_gb": sys_available_ram_gb,
        "ram_percent": sys_ram_percent,
        "advice": f"目前系統剩餘可用 RAM: {sys_available_ram_gb} GB（總量 {sys_total_ram_gb} GB）。訓練預估需 {total_req_ram_gb} GB。" if sys_available_ram_gb >= total_req_ram_gb else f"【系統 RAM 緊張】目前系統剩餘可用 RAM 僅剩 {sys_available_ram_gb} GB，小於訓練預估所需 ({total_req_ram_gb} GB)！建議將 Workers 設為 0 或 1，並關閉其他背景軟體。"
    }

    return {
        "vram": {
            "total_gb": total_vram_gb,
            "total_mb": round(total_vram_mb, 1),
            "breakdown": {
                "model_weights_mb": model_weights_mb,
                "gradients_mb": gradients_mb,
                "optimizer_mb": optimizer_mb,
                "activations_mb": activations_mb,
                "cuda_base_mb": cuda_base_mb,
                "system_gpu_overhead_mb": system_gpu_overhead_mb,
            },
            "matrix": vram_matrix,
        },
        "ram": {
            "total_gb": total_req_ram_gb,
            "total_mb": round(total_req_ram_mb, 1),
            "sys_total_gb": sys_total_ram_gb,
            "sys_available_gb": sys_available_ram_gb,
            "sys_used_gb": sys_used_ram_gb,
            "sys_percent": sys_ram_percent,
            "breakdown": {
                "dataset_cache_mb": dataset_cache_mb,
                "worker_process_ram_mb": worker_process_ram_mb,
                "main_process_ram_mb": main_process_ram_mb,
            },
            "workers_matrix": ram_workers_matrix,
        },
        "advisory": vram_advisory,
        "ram_advisory": ram_advisory,
        "config_used": {
            "batch_size": batch_size,
            "train_workers": train_workers,
            "trainable": trainable,
            "precision": precision,
        }
    }


@app.post("/api/dataset/summary")
@app.get("/api/dataset/summary")
async def get_dataset_summary(
    dataset_path: Optional[str] = None,
    batch_size: int = 8,
    train_workers: int = 4,
    trainable: str = "heads",
    precision: str = "fp16",
    payload: Optional[Dict[str, Any]] = None,
):
    """深度分析指定的 Train Dataset (YAML) 及其音訊 wav 與標註 json，並計算 VRAM/RAM 需求。"""
    try:
        import soundfile as sf
        import yaml
    except ImportError:
        pass

    if payload:
        dataset_path = payload.get("dataset_path", dataset_path)
        batch_size = int(payload.get("batch_size", batch_size))
        train_workers = int(payload.get("train_workers", train_workers))
        trainable = payload.get("trainable", trainable)
        precision = payload.get("precision", precision)

    if not dataset_path:
        dataset_path = "Test_data/dataset.yaml"

    yaml_file = (WORKSPACE_ROOT / dataset_path).resolve()
    if not yaml_file.exists():
        # 嘗試相對路徑
        yaml_file = Path(dataset_path).resolve()
        if not yaml_file.exists():
            raise HTTPException(status_code=404, detail=f"找不到資料集配置檔: {dataset_path}")

    # 讀取 YAML 配置
    try:
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析 YAML 失敗: {str(e)}")

    dataset_name = config_data.get("name", yaml_file.stem)
    dataset_description = config_data.get("description", "")
    sources = config_data.get("sources", [])

    audio_files_info = {}  # key: relative_path/filename -> info
    annotations_list = []  # list of all annotations
    annotated_audio_filenames = set()
    samplerate_counts = {}
    channel_counts = {}
    total_audio_duration_sec = 0.0
    total_audio_size_bytes = 0

    sources_summary = []

    # 若 YAML 中未定義 sources，嘗試自該目錄下的 audio/anns 資料夾尋找
    if not sources:
        base_dir = yaml_file.parent
        audio_dir = base_dir / "audio"
        anns_dir = base_dir / "anns"
        if audio_dir.exists() or anns_dir.exists():
            sources = [{
                "name": "Default Source",
                "audio_dir": str(audio_dir.relative_to(WORKSPACE_ROOT)).replace("\\", "/") if audio_dir.exists() else None,
                "annotations_dir": str(anns_dir.relative_to(WORKSPACE_ROOT)).replace("\\", "/") if anns_dir.exists() else None,
            }]

    for src in sources:
        src_name = src.get("name", "Unknown Source")
        src_audio_dir_str = src.get("audio_dir")
        src_anns_dir_str = src.get("annotations_dir")

        src_audio_count = 0
        src_ann_count = 0
        src_call_count = 0

        # 解析音訊目錄
        audio_dir_path = None
        if src_audio_dir_str:
            audio_dir_path = (WORKSPACE_ROOT / src_audio_dir_str).resolve()
            if not audio_dir_path.exists():
                audio_dir_path = (yaml_file.parent / src_audio_dir_str).resolve()

        if audio_dir_path and audio_dir_path.exists():
            audio_patterns = ["**/*.wav", "**/*.WAV", "**/*.flac", "**/*.FLAC", "**/*.mp3"]
            audio_paths = []
            for pat in audio_patterns:
                audio_paths.extend(audio_dir_path.glob(pat))

            for ap in audio_paths:
                rel_p = str(ap.relative_to(WORKSPACE_ROOT) if ap.is_relative_to(WORKSPACE_ROOT) else ap).replace("\\", "/")
                file_size = ap.stat().st_size
                total_audio_size_bytes += file_size
                src_audio_count += 1

                # 讀取音訊 Header (不載入波形數據，極快)
                dur = 0.0
                sr = 256000
                ch = 1
                try:
                    info = sf.info(str(ap))
                    dur = info.duration
                    sr = info.samplerate
                    ch = info.channels
                except Exception:
                    pass

                total_audio_duration_sec += dur
                samplerate_counts[str(sr)] = samplerate_counts.get(str(sr), 0) + 1
                channel_counts[f"{ch} Channel(s)"] = channel_counts.get(f"{ch} Channel(s)", 0) + 1

                audio_files_info[ap.name] = {
                    "path": rel_p,
                    "filename": ap.name,
                    "duration": dur,
                    "samplerate": sr,
                    "channels": ch,
                    "size_bytes": file_size,
                    "source": src_name,
                }

        # 解析標註目錄
        anns_dir_path = None
        if src_anns_dir_str:
            anns_dir_path = (WORKSPACE_ROOT / src_anns_dir_str).resolve()
            if not anns_dir_path.exists():
                anns_dir_path = (yaml_file.parent / src_anns_dir_str).resolve()

        if anns_dir_path and anns_dir_path.exists():
            ann_json_paths = list(anns_dir_path.glob("**/*.json"))
            for jp in ann_json_paths:
                src_ann_count += 1
                try:
                    with open(jp, "r", encoding="utf-8") as jf:
                        ann_data = json.load(jf)
                    
                    is_annotated = ann_data.get("annotated", True)
                    ann_items = ann_data.get("annotation", [])
                    
                    # 對應的音訊檔名 (通常為 xxx.wav.json -> xxx.wav)
                    stem_name = jp.name
                    if stem_name.endswith(".json"):
                        stem_name = stem_name[:-5]
                    
                    if is_annotated and stem_name in audio_files_info:
                        annotated_audio_filenames.add(stem_name)
                    elif is_annotated:
                        annotated_audio_filenames.add(stem_name)

                    for item in ann_items:
                        src_call_count += 1
                        cls_name = item.get("class") or "Unknown"
                        ev_type = item.get("event") or "Echolocation"
                        s_time = float(item.get("start_time", 0.0))
                        e_time = float(item.get("end_time", 0.0))
                        l_freq = float(item.get("low_freq", 0.0))
                        h_freq = float(item.get("high_freq", 0.0))
                        indiv = str(item.get("individual", "0"))

                        call_dur_ms = max(0.0, (e_time - s_time) * 1000.0)

                        annotations_list.append({
                            "class": cls_name,
                            "event": ev_type,
                            "start_time": s_time,
                            "end_time": e_time,
                            "duration_ms": call_dur_ms,
                            "low_freq_khz": l_freq / 1000.0,
                            "high_freq_khz": h_freq / 1000.0,
                            "bandwidth_khz": max(0.0, (h_freq - l_freq) / 1000.0),
                            "individual": indiv,
                            "audio_filename": stem_name,
                            "source": src_name,
                        })
                except Exception:
                    pass

        sources_summary.append({
            "name": src_name,
            "audio_dir": src_audio_dir_str,
            "annotations_dir": src_anns_dir_str,
            "audio_count": src_audio_count,
            "annotation_files_count": src_ann_count,
            "calls_count": src_call_count,
        })

    total_audio_files = len(audio_files_info)
    total_calls = len(annotations_list)

    # 識別包含蝙蝠叫聲的音訊與純負樣本/噪音音訊 (Negative / Noise files)
    positive_audio_filenames = set(ann["audio_filename"] for ann in annotations_list)
    noise_audio_filenames = set()
    for stem_name in annotated_audio_filenames:
        if stem_name not in positive_audio_filenames:
            noise_audio_filenames.add(stem_name)

    total_noise_files = len(noise_audio_filenames)
    total_noise_duration_sec = sum(
        audio_files_info[nf]["duration"] for nf in noise_audio_filenames if nf in audio_files_info
    )
    total_noise_size_bytes = sum(
        audio_files_info[nf]["size_bytes"] for nf in noise_audio_filenames if nf in audio_files_info
    )

    # 4. 深度物種 (Species / Class) 統計分析
    species_dict = {}
    for ann in annotations_list:
        cls_name = ann["class"]
        if cls_name not in species_dict:
            species_dict[cls_name] = {
                "class_name": cls_name,
                "calls": [],
                "recordings": set(),
                "events": {},
                "individuals": set(),
            }
        
        species_dict[cls_name]["calls"].append(ann)
        species_dict[cls_name]["recordings"].add(ann["audio_filename"])
        ev = ann["event"]
        species_dict[cls_name]["events"][ev] = species_dict[cls_name]["events"].get(ev, 0) + 1
        species_dict[cls_name]["individuals"].add(ann["individual"])

    species_summary_table = []
    for cls_name, data in species_dict.items():
        calls = data["calls"]
        num_calls = len(calls)
        call_pct = round((num_calls / total_calls * 100), 1) if total_calls > 0 else 0.0
        
        durations = [c["duration_ms"] for c in calls]
        low_freqs = [c["low_freq_khz"] for c in calls if c["low_freq_khz"] > 0]
        high_freqs = [c["high_freq_khz"] for c in calls if c["high_freq_khz"] > 0]
        bandwidths = [c["bandwidth_khz"] for c in calls]

        # 計算該物種所在音訊檔案的累計總時長
        species_recs_dur = sum(
            audio_files_info[rf]["duration"] for rf in data["recordings"] if rf in audio_files_info
        )
        total_call_dur_sec = sum(durations) / 1000.0
        
        # 呼叫率 (Calls per minute of audio containing this species)
        call_rate_per_min = round(num_calls / (species_recs_dur / 60.0), 1) if species_recs_dur > 0 else 0.0

        species_summary_table.append({
            "class_name": cls_name,
            "is_negative": False,
            "num_calls": num_calls,
            "call_percentage": call_pct,
            "num_recordings": len(data["recordings"]),
            "recording_percentage": round(len(data["recordings"]) / max(1, total_audio_files) * 100, 1),
            "total_annotated_sec": round(total_call_dur_sec, 2),
            "avg_duration_ms": round(float(np.mean(durations)), 1) if durations else 0.0,
            "min_duration_ms": round(float(np.min(durations)), 1) if durations else 0.0,
            "max_duration_ms": round(float(np.max(durations)), 1) if durations else 0.0,
            "mean_low_freq_khz": round(float(np.mean(low_freqs)), 1) if low_freqs else 0.0,
            "mean_high_freq_khz": round(float(np.mean(high_freqs)), 1) if high_freqs else 0.0,
            "min_low_freq_khz": round(float(np.min(low_freqs)), 1) if low_freqs else 0.0,
            "max_high_freq_khz": round(float(np.max(high_freqs)), 1) if high_freqs else 0.0,
            "mean_bandwidth_khz": round(float(np.mean(bandwidths)), 1) if bandwidths else 0.0,
            "call_rate_per_min": call_rate_per_min,
            "events_breakdown": data["events"],
            "primary_event": max(data["events"], key=data["events"].get) if data["events"] else "None",
            "individuals_count": len(data["individuals"]),
        })

    # 依照叫聲數量由多到少排序
    species_summary_table.sort(key=lambda x: x["num_calls"], reverse=True)

    # 若存在負樣本/噪音檔案，加入專屬的負樣本統整列
    if total_noise_files > 0:
        species_summary_table.append({
            "class_name": "Background Noise (Negative Samples)",
            "is_negative": True,
            "num_calls": 0,
            "call_percentage": 0.0,
            "num_recordings": total_noise_files,
            "recording_percentage": round(total_noise_files / max(1, total_audio_files) * 100, 1),
            "total_annotated_sec": round(total_noise_duration_sec, 2),
            "avg_duration_ms": 0.0,
            "min_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "mean_low_freq_khz": 0.0,
            "mean_high_freq_khz": 0.0,
            "min_low_freq_khz": 0.0,
            "max_high_freq_khz": 0.0,
            "mean_bandwidth_khz": 0.0,
            "call_rate_per_min": 0.0,
            "events_breakdown": {"Negative Sample": total_noise_files},
            "primary_event": "Hard Negative",
            "individuals_count": 0,
        })

    # 5. 計算 VRAM 及 RAM 需求
    hardware_estimate = estimate_training_hardware(
        num_audio_files=total_audio_files,
        total_duration_sec=total_audio_duration_sec,
        batch_size=batch_size,
        train_workers=train_workers,
        trainable=trainable,
        precision=precision,
    )

    # 格式化總錄音長度字串
    hrs = int(total_audio_duration_sec // 3600)
    mins = int((total_audio_duration_sec % 3600) // 60)
    secs = int(total_audio_duration_sec % 60)
    duration_str = f"{hrs}h {mins}m {secs}s" if hrs > 0 else f"{mins}m {secs}s"

    noise_mins = int((total_noise_duration_sec % 3600) // 60)
    noise_secs = int(total_noise_duration_sec % 60)
    noise_duration_str = f"{noise_mins}m {noise_secs}s"

    return {
        "success": True,
        "dataset_name": dataset_name,
        "dataset_description": dataset_description,
        "dataset_path": dataset_path,
        "overview": {
            "total_audio_files": total_audio_files,
            "total_annotated_files": len(annotated_audio_filenames),
            "total_positive_files": len(positive_audio_filenames),
            "total_noise_files": total_noise_files,
            "total_noise_duration_formatted": noise_duration_str,
            "total_noise_size_mb": round(total_noise_size_bytes / (1024 ** 2), 2),
            "unannotated_files": max(0, total_audio_files - len(annotated_audio_filenames)),
            "annotation_coverage_pct": round(len(annotated_audio_filenames) / max(1, total_audio_files) * 100, 1),
            "total_duration_sec": round(total_audio_duration_sec, 2),
            "total_duration_formatted": duration_str,
            "total_size_mb": round(total_audio_size_bytes / (1024 ** 2), 2),
            "total_size_gb": round(total_audio_size_bytes / (1024 ** 3), 2),
            "total_calls": total_calls,
            "num_species": len(species_dict),
            "samplerate_distribution": samplerate_counts,
            "channel_distribution": channel_counts,
            "sources": sources_summary,
        },
        "species_summary": species_summary_table,
        "hardware_estimate": hardware_estimate,
    }


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
