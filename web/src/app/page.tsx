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
import { Activity, Zap, ShieldCheck, Waves, FileAudio, Clock } from 'lucide-react';

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
    <div className="min-h-screen bg-[#121214] text-zinc-100 flex flex-col selection:bg-emerald-500/25 selection:text-emerald-300">
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
          <section className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-300">
            {/* Quick Metrics Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
              <div className="p-4 rounded-2xl bg-[#18181b] border border-[#27272a] shadow-lg flex flex-col justify-between">
                <div className="flex items-center gap-1.5 text-zinc-400 text-xs font-medium mb-1">
                  <FileAudio className="w-3.5 h-3.5 text-zinc-500" />
                  <span>音訊檔案名稱</span>
                </div>
                <span className="text-sm font-semibold text-zinc-200 truncate font-mono" title={result.filename}>
                  {result.filename}
                </span>
              </div>

              <div className="p-4 rounded-2xl bg-[#18181b] border border-[#27272a] shadow-lg flex flex-col justify-between">
                <div className="flex items-center gap-1.5 text-zinc-400 text-xs font-medium mb-1">
                  <Waves className="w-3.5 h-3.5 text-emerald-500/70" />
                  <span>時長 / 採樣率</span>
                </div>
                <span className="text-sm font-mono font-semibold text-emerald-400">
                  {result.duration.toFixed(2)}s <span className="text-xs text-zinc-400 font-normal">@ {(result.sample_rate / 1000).toFixed(0)}kHz</span>
                </span>
              </div>

              <div className="p-4 rounded-2xl bg-[#18181b] border border-[#27272a] shadow-lg flex flex-col justify-between">
                <div className="flex items-center gap-1.5 text-zinc-400 text-xs font-medium mb-1">
                  <Activity className="w-3.5 h-3.5 text-cyan-500/70" />
                  <span>定位叫聲數 (&ge;{Math.round(threshold * 100)}%)</span>
                </div>
                <span className="text-sm font-mono font-bold text-cyan-400">
                  {result.detections.filter((d) => d.detection_score >= threshold).length} <span className="text-xs text-zinc-400 font-normal">/ {result.detections.length} 次</span>
                </span>
              </div>

              <div className="p-4 rounded-2xl bg-[#18181b] border border-[#27272a] shadow-lg flex flex-col justify-between">
                <div className="flex items-center gap-1.5 text-zinc-400 text-xs font-medium mb-1">
                  <Clock className="w-3.5 h-3.5 text-purple-500/70" />
                  <span>端側推論耗時</span>
                </div>
                <span className="text-sm font-mono font-semibold text-purple-400">
                  {result.inference_time_ms} ms <span className="text-[11px] text-zinc-400 font-normal">({backend.toUpperCase()})</span>
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
            <div className="p-6 rounded-2xl bg-[#18181b] border border-[#27272a] space-y-2.5 hover:border-[#3f3f46] transition-colors shadow-md">
              <div className="w-10 h-10 rounded-xl bg-[#121214] border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-inner">
                <Zap className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-zinc-200">WebAssembly + WebGPU 核心</h4>
              <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                利用 ONNX Runtime Web 直接調用使用者的本機 GPU 或 CPU SIMD 進行即時神經網路推論，毫秒級產出分析結果。
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-[#18181b] border border-[#27272a] space-y-2.5 hover:border-[#3f3f46] transition-colors shadow-md">
              <div className="w-10 h-10 rounded-xl bg-[#121214] border border-cyan-500/30 flex items-center justify-center text-cyan-400 shadow-inner">
                <Activity className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-zinc-200">500kHz 超音波高解析支援</h4>
              <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                自行封裝 PCM WAV 解碼器，杜絕瀏覽器強制降採樣問題，完整保留高達 128kHz+ 的蝙蝠超音波聲學特徵。
              </p>
            </div>

            <div className="p-6 rounded-2xl bg-[#18181b] border border-[#27272a] space-y-2.5 hover:border-[#3f3f46] transition-colors shadow-md">
              <div className="w-10 h-10 rounded-xl bg-[#121214] border border-purple-500/30 flex items-center justify-center text-purple-400 shadow-inner">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <h4 className="text-sm font-semibold text-zinc-200">100% 離線隱私安全</h4>
              <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                可純靜態託管於 Vercel / GitHub Pages。音訊檔案完全不離開您的電腦，極端保障野外生態調查資料的機密性。
              </p>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#27272a] py-6 px-6 text-center text-xs text-zinc-500 space-y-1 mt-auto">
        <p className="font-mono">BatDetect2 In-Browser Studio • Powered by Next.js & ONNX Runtime Web</p>
        <p className="text-[11px] text-zinc-600">
          支援一鍵部署至 Vercel (Static Export) • 零後端維護成本
        </p>
      </footer>
    </div>
  );

}

