# BatDetect2 Web (Next.js + WASM / WebGPU)

一個基於 **Next.js (TypeScript) + ONNX Runtime Web** 的純前端超音波蝙蝠聲音偵測與物種辨識工作台。

## ✨ 特色亮點

- 🚀 **100% 本地瀏覽器端推論**：無需伺服器後端或 GPU 伺服器，音訊完全不上傳，零伺服器維護成本、保障隱私。
- ⚡ **WASM + WebGPU 硬體加速**：利用 WebAssembly SIMD 多執行緒與 WebGPU，實現毫秒級神經網路運算。
- 🦇 **500kHz 超音波高取樣率支援**：純前端 PCM WAV 解碼，避免瀏覽器強制降採樣，完整保留 10kHz ~ 128kHz 超音波特徵。
- 🎨 **互動式頻譜工作台**：支援 Magma/Viridis/Inferno 色階、即時叫聲標籤框（Bounding Box）、信心度門檻調整與縮放檢視。
- 🔊 **超音波 0.1x 降速聆聽**：內建 Time Expansion 音訊播放器，將人耳聽不見的超音波按比例降至可聽頻率範圍。
- 📊 **資料匯出**：支援將偵測結果（時間點、頻率範圍、物種、信心度）一鍵匯出為 **CSV** 或 **JSON** 格式。

---

## 🛠️ 本地開發 (Local Development)

```bash
cd web
npm install
npm run dev
```

開啟瀏覽器造訪 [http://localhost:3000](http://localhost:3000)。

---

## 🚀 部署至 Vercel (Deploy to Vercel)

本專案支援靜態輸出 (`output: 'export'`)，可直接一鍵部署至 Vercel：

1. 將專案 Push 至您的 GitHub Repository。
2. 登入 [Vercel](https://vercel.com/)，點選 **Add New Project**。
3. 選擇您的 Repository。
4. **Root Directory** 設定為 `web`。
5. 點擊 **Deploy**，幾秒鐘內即可完成全域 CDN 部署！

---

## 🔄 更換 / 重新導出新的 Checkpoint 模型

若日後訓練了新的 PyTorch Checkpoint（如 `last.ckpt`）：

```bash
# 在專案根目錄執行 (使用 uv)
uv run python scripts/export_onnx.py "outputs/checkpoints/your_model/last.ckpt"
```

此指令會自動將模型轉換為 `web/public/models/batdetect2.onnx` 並同步更新 `metadata.json`。
