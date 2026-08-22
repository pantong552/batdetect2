'use client';

import React from 'react';
import { Activity, Cpu, Sparkles, ShieldCheck } from 'lucide-react';
import { ModelMetadata } from '../lib/types';

interface HeaderProps {
  metadata: ModelMetadata | null;
  backend: 'webgpu' | 'wasm';
  isReady: boolean;
}

export const Header: React.FC<HeaderProps> = ({ metadata, backend, isReady }) => {
  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 px-6 py-3.5 flex items-center justify-between shadow-lg">
      <div className="flex items-center gap-3.5">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 via-teal-500/20 to-cyan-500/20 border border-emerald-500/40 text-emerald-400">
          <Activity className="w-5 h-5 animate-pulse" />
          <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
          </span>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
              BatDetect2 <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-mono">Web (WASM / WebGPU)</span>
            </h1>
          </div>
          <p className="text-xs text-slate-400">純前端超音波蝙蝠聲音偵測與物種辨識工作台</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Hardware Backend Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-700/60 text-xs">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-400">硬體加速:</span>
          <span className={`font-mono font-semibold uppercase ${backend === 'webgpu' ? 'text-emerald-400' : 'text-cyan-400'}`}>
            {backend}
          </span>
        </div>

        {/* Target Species Badges */}
        {metadata && (
          <div className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-700/60 text-xs">
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-slate-400">辨識物種:</span>
            <div className="flex gap-1">
              {metadata.classes.map((cls) => (
                <span key={cls} className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[11px] font-mono border border-slate-700">
                  {cls}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Zero-Server Privacy Badge */}
        <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-950/40 border border-emerald-800/40 text-xs text-emerald-300">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span>100% 本地運算 (零上傳)</span>
        </div>
      </div>
    </header>
  );
};
