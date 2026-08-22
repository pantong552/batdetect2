'use client';

import React, { useRef, useState } from 'react';
import { UploadCloud, FileAudio, PlayCircle, Loader2, Sparkles } from 'lucide-react';

interface FileUploaderProps {
  onFileSelected: (file: File | ArrayBuffer, filename: string) => void;
  isLoading: boolean;
  progressText: string;
  progressPct: number;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  onFileSelected,
  isLoading,
  progressText,
  progressPct,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.name.toLowerCase().endsWith('.wav')) {
        onFileSelected(file, file.name);
      } else {
        alert('請上傳 .wav 格式的蝙蝠錄音檔');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelected(files[0], files[0].name);
    }
  };

  const handleTryDemo = async () => {
    try {
      const res = await fetch('/demo/demo_bat.wav');
      if (!res.ok) throw new Error('Demo file not found');
      const buffer = await res.arrayBuffer();
      onFileSelected(buffer, '20250502_190540_HM.wav');
    } catch (err) {
      alert(`載入範例音檔失敗: ${err}`);
    }
  };

  return (
    <div className="w-full">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
        className={`relative group cursor-pointer transition-all duration-300 rounded-2xl border-2 border-dashed p-8 text-center flex flex-col items-center justify-center ${
          isDragOver
            ? 'border-emerald-400 bg-emerald-950/20 scale-[1.01]'
            : 'border-slate-700/70 hover:border-emerald-500/60 bg-slate-900/40 hover:bg-slate-900/70'
        } ${isLoading ? 'pointer-events-none opacity-80' : ''}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,audio/wav,audio/x-wav"
          className="hidden"
          onChange={handleFileChange}
        />

        {isLoading ? (
          <div className="flex flex-col items-center gap-4 py-4 w-full max-w-md">
            <div className="relative flex items-center justify-center w-16 h-16 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              <Loader2 className="w-8 h-8 animate-spin" />
            </div>
            <div className="w-full">
              <div className="flex justify-between text-xs mb-1.5 font-medium text-slate-300">
                <span>{progressText}</span>
                <span className="font-mono text-emerald-400">{progressPct}%</span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                <div
                  className="h-full bg-gradient-to-r from-teal-500 to-emerald-400 transition-all duration-300 rounded-full"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-b from-slate-800 to-slate-900 border border-slate-700 flex items-center justify-center text-emerald-400 group-hover:scale-110 group-hover:text-emerald-300 transition-transform shadow-inner">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-slate-200">
                拖曳 <span className="text-emerald-400">.wav</span> 蝙蝠錄音檔至此，或點擊選擇檔案
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                支援 16/24/32-bit PCM，保留 256k~500kHz 高頻超音波（100% 在瀏覽器內運算）
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Demo Audio Button */}
      {!isLoading && (
        <div className="flex items-center justify-center gap-2 mt-3">
          <span className="text-xs text-slate-500">沒有音檔？</span>
          <button
            onClick={handleTryDemo}
            type="button"
            className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium text-emerald-300 bg-emerald-950/40 hover:bg-emerald-900/60 border border-emerald-800/50 rounded-lg transition-colors shadow-sm"
          >
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            一鍵試用內建範例錄音 (Demo WAV)
          </button>
        </div>
      )}
    </div>
  );
};
