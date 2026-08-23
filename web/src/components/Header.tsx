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
    <header className="sticky top-0 z-50 bg-[#18181b]/90 border-b border-[#27272a] px-6 py-3.5 flex items-center justify-between shadow-2xl backdrop-blur-md">
      <div className="flex items-center gap-3.5">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-[#27272a] border border-emerald-500/30 text-emerald-400 shadow-inner">
          <Activity className="w-5 h-5 animate-pulse" />
          <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
          </span>
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base sm:text-lg font-bold tracking-tight text-zinc-100 flex items-center gap-2 font-mono">
              BatDetect2 <span className="text-[11px] font-medium px-2 py-0.5 rounded-md bg-[#27272a] text-emerald-400 border border-emerald-500/30 font-mono tracking-normal">Web Lab</span>
            </h1>
          </div>
          <p className="text-xs text-zinc-400">純前端超音波蝙蝠聲音偵測與物種辨識工作台</p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Hardware Backend Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#27272a] border border-[#3f3f46] text-xs shadow-sm">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-zinc-400 font-medium">加速核心:</span>
          <span className={`font-mono font-semibold uppercase ${backend === 'webgpu' ? 'text-emerald-400' : 'text-cyan-400'}`}>
            {backend}
          </span>
        </div>

        {/* Target Species Badges */}
        {metadata && (
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#27272a] border border-[#3f3f46] text-xs shadow-sm">
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-zinc-400 font-medium">涵蓋物種:</span>
            <div className="flex gap-1.5">
              {metadata.classes.map((cls) => (
                <span key={cls} className="px-1.5 py-0.5 rounded bg-[#18181b] text-zinc-300 text-[11px] font-mono border border-[#3f3f46]">
                  {cls}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Zero-Server Privacy Badge */}
        <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#27272a] border border-emerald-500/30 text-xs text-emerald-300 shadow-sm">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
          <span className="font-medium">100% 離線端運算</span>
        </div>
      </div>
    </header>
  );
};


