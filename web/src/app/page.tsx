'use client';

import React, { useState, useEffect } from 'react';
import confetti from 'canvas-confetti';
import { Header } from '../components/Header';
import { FileUploader } from '../components/FileUploader';
import { SpectrogramViewer } from '../components/SpectrogramViewer';
import { AudioPlayer } from '../components/AudioPlayer';
import { DetectionTable } from '../components/DetectionTable';
import { batDetector } from '../lib/inference/detector';
import { AudioAnalysisResult, Detection, ModelMetadata } from '../lib/types';
import { Activity, Sparkles, Zap, ShieldCheck, HelpCircle } from 'lucide-react';

export default function Home() {
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [backend, setBackend] = useState<'webgpu' | 'wasm'>('wasm');
  const [isModelReady, setIsModelReady] = useState(false);

  // Analysis State
  const [isLoading, setIsLoading] = useState(false);
  const [progressText, setProgressText] = useState('準備中...');
  const [progressPct, setProgressPct] = useState(0);

  // Result State
  const [currentBuffer, setCurrentBuffer] = useState<ArrayBuffer | null>(null);
  const [result, setResult] = useState<AudioAnalysisResult | null>(null);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);
  const [threshold, setThreshold] = useState<number>(0.2);

  // Initialize model on mount
  useEffect(() => {
    batDetector
      .initialize((pct, msg) => {
        setProgressPct(pct);
        setProgressText(msg);
      })
      .then(() => {
        setMetadata(batDetector.getMetadata());
        setBackend(batDetector.getBackend());
        setIsModelReady(true);
      })
      .catch((err) => {
        console.error('Model initialization error:', err);
      });
  }, []);

  // Handle file selection and start browser-side inference
  const handleFileSelected = async (fileOrBuffer: File | ArrayBuffer, filename: string) => {
    setIsLoading(true);
    setProgressPct(5);
    setProgressText('讀取音訊檔案...');
    setSelectedDetection(null);
    setCurrentTime(0);

    try {
      let buffer: ArrayBuffer;
      if (fileOrBuffer instanceof File) {
        buffer = await fileOrBuffer.arrayBuffer();
      } else {
        buffer = fileOrBuffer;
      }
      setCurrentBuffer(buffer);

      const analysis = await batDetector.analyzeAudio(
        buffer,
        filename,
        threshold,
        (pct, step) => {
          setProgressPct(pct);
          setProgressText(step);
        }
      );

      setResult(analysis);

      // Trigger celebratory micro-animation
      if (analysis.detections.length > 0) {
        confetti({
          particleCount: 40,
          spread: 60,
          origin: { y: 0.85 },
          colors: ['#10b981', '#06b6d4', '#8b5cf6'],
        });
      }
    } catch (err: unknown) {
      console.error(err);
      const errMsg = err instanceof Error ? err.message : String(err);
      alert(`分析失敗: ${errMsg}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#07090e] text-slate-100 flex flex-col selection:bg-emerald-500/30 selection:text-emerald-300">
      {/* Top Navigation */}
      <Header metadata={metadata} backend={backend} isReady={isModelReady} />

      {/* Main Workspace Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {/* Hero Banner / File Uploader */}
        <section className="space-y-4">
          <FileUploader
            onFileSelected={handleFileSelected}
            isLoading={isLoading}
            progressText={progressText}
            progressPct={progressPct}
          />
        </section>

        {/* Results Studio View */}
        {result && (
          <section className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Quick Metrics Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col">
                <span className="text-[11px] text-slate-400 font-medium">檔案名稱</span>
                <span className="text-sm font-semibold text-slate-200 truncate" title={result.filename}>
                  {result.filename}
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col">
                <span className="text-[11px] text-slate-400 font-medium">音訊長度 / 採樣率</span>
                <span className="text-sm font-mono font-semibold text-emerald-400">
                  {result.duration.toFixed(2)}s @ {(result.sample_rate / 1000).toFixed(0)}kHz
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col">
                <span className="text-[11px] text-slate-400 font-medium">偵測叫聲數 (&ge;{Math.round(threshold * 100)}%)</span>
                <span className="text-sm font-mono font-bold text-cyan-400">
                  {result.detections.filter((d) => d.detection_score >= threshold).length} / {result.detections.length} 次
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col">
                <span className="text-[11px] text-slate-400 font-medium">本地推論耗時</span>
                <span className="text-sm font-mono font-semibold text-purple-400">
                  {result.inference_time_ms} ms ({backend.toUpperCase()})
                </span>
              </div>
            </div>

            {/* Interactive Spectrogram Viewer */}
            <SpectrogramViewer
              result={result}
              currentTime={currentTime}
              onSeek={(t) => setCurrentTime(t)}
              selectedDetection={selectedDetection}
              onSelectDetection={(d) => setSelectedDetection(d)}
              threshold={threshold}
              onThresholdChange={(th) => setThreshold(th)}
            />

            {/* Ultrasound Time Expansion Player */}
            <AudioPlayer
              audioBuffer={currentBuffer}
              sampleRate={result.sample_rate}
              duration={result.duration}
              currentTime={currentTime}
              onTimeUpdate={(t) => setCurrentTime(t)}
            />

            {/* Detections List & Statistics */}
            <DetectionTable
              result={result}
              selectedDetection={selectedDetection}
              onSelectDetection={(d) => {
                setSelectedDetection(d);
                setCurrentTime(d.start_time);
              }}
              threshold={threshold}
              onThresholdChange={(th) => setThreshold(th)}
            />
          </section>
        )}

        {/* Feature Highlights (when no file is loaded) */}
        {!result && !isLoading && (
          <section className="grid sm:grid-cols-3 gap-4 pt-4">
            <div className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800/80 space-y-2">
              <div className="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <Zap className="w-4 h-4" />
              </div>
              <h4 className="text-sm font-semibold text-slate-200">WebAssembly + WebGPU 加速</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                利用 ONNX Runtime Web 直接調用使用者的本機 GPU 或 CPU SIMD 進行即時神經網路推論，毫秒級產出分析結果。
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800/80 space-y-2">
              <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <Activity className="w-4 h-4" />
              </div>
              <h4 className="text-sm font-semibold text-slate-200">500kHz 超音波無損解析</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                自行封裝 PCM WAV 解碼器，杜絕瀏覽器強制降採樣問題，完整保留高達 128kHz+ 的蝙蝠超音波特徵。
              </p>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/40 border border-slate-800/80 space-y-2">
              <div className="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <h4 className="text-sm font-semibold text-slate-200">零伺服器 & 100% 隱私安全</h4>
              <p className="text-xs text-slate-400 leading-relaxed">
                可純靜態部署至 Vercel。音訊檔案完全不離開您的電腦，極端保障生態調查資料的隱私與安全性。
              </p>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-6 px-6 text-center text-xs text-slate-500 space-y-1">
        <p>BatDetect2 In-Browser Studio • Powered by Next.js & ONNX Runtime Web</p>
        <p className="text-[11px] text-slate-600">
          支援一鍵部署至 Vercel (Static Export) • 零後端維護成本
        </p>
      </footer>
    </div>
  );
}
