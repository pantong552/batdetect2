'use client';

import React, { useRef, useState } from 'react';
import { UploadCloud, Loader2, Sparkles } from 'lucide-react';

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
        className={`relative group cursor-pointer transition-all duration-200 rounded-2xl border-2 border-dashed p-8 sm:p-10 text-center flex flex-col items-center justify-center ${
          isDragOver
            ? 'border-emerald-500 bg-[#27272a]/90 scale-[1.008] shadow-2xl ring-2 ring-emerald-500/20'
            : 'border-[#27272a] hover:border-[#3f3f46] bg-[#18181b]/90 hover:bg-[#202024]'
        } ${isLoading ? 'pointer-events-none opacity-85' : ''}`}
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
            <div className="relative flex items-center justify-center w-16 h-16 rounded-2xl bg-[#121214] border border-emerald-500/30 text-emerald-400 shadow-inner">
              <Loader2 className="w-8 h-8 animate-spin" />
            </div>
            <div className="w-full">
              <div className="flex justify-between text-xs mb-2 font-medium text-zinc-300">
                <span className="font-mono">{progressText}</span>
                <span className="font-mono text-emerald-400 font-bold">{progressPct}%</span>
              </div>
              <div className="w-full h-2 bg-[#121214] rounded-full overflow-hidden border border-[#27272a]">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300 rounded-full"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-14 h-14 rounded-2xl bg-[#121214] border border-[#27272a] flex items-center justify-center text-emerald-400 group-hover:text-emerald-300 group-hover:border-emerald-500/40 group-hover:scale-105 transition-all shadow-inner">
              <UploadCloud className="w-7 h-7" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-zinc-200">
                拖曳 <span className="text-emerald-400 font-mono">.wav</span> 蝙蝠錄音檔至此，或點擊選擇檔案
              </h3>
              <p className="text-xs text-zinc-400 mt-1.5 font-sans">
                支援 16/24/32-bit PCM，保留 256k~500kHz 高頻超音波（100% 瀏覽器本地端運算）
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Demo Audio Button */}
      {!isLoading && (
        <div className="flex items-center justify-center gap-2 mt-3.5">
          <span className="text-xs text-zinc-400">沒有音檔？</span>
          <button
            onClick={handleTryDemo}
            type="button"
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-medium text-zinc-200 hover:text-emerald-300 bg-[#18181b] hover:bg-[#27272a] border border-[#3f3f46] hover:border-emerald-500/40 rounded-xl transition-all shadow-sm cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            一鍵試用內建範例錄音 (Demo WAV)
          </button>
        </div>
      )}
    </div>
  );
};


