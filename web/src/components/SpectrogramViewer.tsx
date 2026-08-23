'use client';

import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { AudioAnalysisResult, Detection } from '../lib/types';
import { ZoomIn, ZoomOut, RotateCcw, Maximize2, SlidersHorizontal, Activity } from 'lucide-react';

interface SpectrogramViewerProps {
  result: AudioAnalysisResult;
  currentTime: number;
  onSeek: (time: number) => void;
  selectedDetection: Detection | null;
  onSelectDetection: (det: Detection | null) => void;
  threshold?: number;
  onThresholdChange?: (th: number) => void;
}

// Color mapping for bat species
const SPECIES_COLORS: { [key: string]: { border: string; fill: string; text: string } } = {
  hiparm: { border: '#10b981', fill: 'rgba(16, 185, 129, 0.22)', text: '#34d399' }, // Emerald
  myohor: { border: '#06b6d4', fill: 'rgba(6, 182, 212, 0.22)', text: '#22d3ee' }, // Cyan
  pipten: { border: '#a855f7', fill: 'rgba(168, 85, 247, 0.22)', text: '#c084fc' }, // Purple
  tylful: { border: '#f59e0b', fill: 'rgba(245, 158, 11, 0.22)', text: '#fbbf24' }, // Amber
  default: { border: '#ec4899', fill: 'rgba(236, 72, 153, 0.22)', text: '#f472b6' }, // Pink
};

export const SpectrogramViewer: React.FC<SpectrogramViewerProps> = ({
  result,
  currentTime,
  onSeek,
  selectedDetection,
  onSelectDetection,
  threshold = 0.2,
  onThresholdChange,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);

  const [hoveredDetection, setHoveredDetection] = useState<Detection | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number; time: number; freq: number } | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [colormap, setColormap] = useState<'magma' | 'viridis' | 'inferno'>('magma');
  const [viewHeight, setViewHeight] = useState<number>(480);

  const { spectrogram, duration, detections } = result;
  const minFreq = spectrogram.minFreq || 10000;
  const maxFreq = spectrogram.maxFreq || 128000;

  // Filter detections by threshold in real-time (matching Python detection_score)
  const filteredDetections = useMemo(() => {
    return detections.filter((d) => d.detection_score >= threshold);
  }, [detections, threshold]);

  // 1. Render Spectrogram Heatmap Image
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { data, width, height } = spectrogram;
    canvas.width = width;
    canvas.height = height;

    const imgData = ctx.createImageData(width, height);
    const pixels = imgData.data;

    // Calculate robust contrast normalization (percentile 99.5)
    const sampleSize = Math.min(data.length, 50000);
    const sample = new Float32Array(sampleSize);
    const step = Math.max(1, Math.floor(data.length / sampleSize));
    for (let i = 0; i < sampleSize; i++) {
      sample[i] = data[i * step] || 0;
    }
    sample.sort();
    const p99 = sample[Math.floor(sampleSize * 0.995)] || 1.0;
    const p05 = sample[Math.floor(sampleSize * 0.05)] || 0.0;
    const range = Math.max(1e-5, p99 - p05);

    // Draw bottom-up (low frequency at bottom y=0 -> displayed at bottom)
    for (let y = 0; y < height; y++) {
      const srcY = height - 1 - y;
      const rowOffset = srcY * width;

      for (let x = 0; x < width; x++) {
        const val = data[rowOffset + x];
        let norm = (val - p05) / range;
        norm = Math.min(1.0, Math.max(0.0, norm));
        norm = Math.pow(norm, 0.85);

        const [r, g, b] = getColormapRGB(norm, colormap);
        const idx = (y * width + x) * 4;
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [spectrogram, colormap]);

  // 2. Draw Overlays (Bounding Boxes + Playhead)
  const renderOverlay = useCallback(() => {
    const overlay = overlayCanvasRef.current;
    const container = containerRef.current;
    if (!overlay || !container) return;

    const containerWidth = container.clientWidth || 800;
    const targetWidth = Math.round(containerWidth * zoomLevel);
    const targetHeight = viewHeight;

    if (overlay.width !== targetWidth || overlay.height !== targetHeight) {
      overlay.width = targetWidth;
      overlay.height = targetHeight;
    }

    const ctx = overlay.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, targetWidth, targetHeight);

    // Draw Filtered Bounding Boxes
    for (const det of filteredDetections) {
      const isSelected = selectedDetection?.id === det.id;
      const isHovered = hoveredDetection?.id === det.id;

      // Coordinate mapping (Time -> X, Freq -> Y)
      const x0 = (det.start_time / duration) * targetWidth;
      const x1 = (det.end_time / duration) * targetWidth;
      const boxW = Math.max(x1 - x0, 10);

      // Y-axis: 0 is top (maxFreq), targetHeight is bottom (minFreq)
      const y0 = targetHeight - ((det.max_freq - minFreq) / (maxFreq - minFreq)) * targetHeight;
      const y1 = targetHeight - ((det.min_freq - minFreq) / (maxFreq - minFreq)) * targetHeight;
      const boxH = Math.max(y1 - y0, 12);

      const color = SPECIES_COLORS[det.species] || SPECIES_COLORS.default;

      // Fill & Stroke
      ctx.fillStyle = isSelected || isHovered ? color.fill.replace('0.22', '0.45') : color.fill;
      ctx.fillRect(x0, y0, boxW, boxH);

      ctx.strokeStyle = color.border;
      ctx.lineWidth = isSelected || isHovered ? 2.5 : 1.5;
      ctx.setLineDash(isSelected ? [4, 2] : []);
      ctx.strokeRect(x0, y0, boxW, boxH);
      ctx.setLineDash([]);

      // Label Tag
      const label = `${det.species} (${Math.round(det.confidence * 100)}%)`;
      ctx.font = '11px monospace';
      const textMetrics = ctx.measureText(label);
      const tagW = textMetrics.width + 8;
      const tagH = 16;

      ctx.fillStyle = 'rgba(11, 15, 25, 0.92)';
      ctx.fillRect(x0, Math.max(0, y0 - tagH), tagW, tagH);

      ctx.fillStyle = color.text;
      ctx.fillText(label, x0 + 4, Math.max(12, y0 - 3));
    }

    // Draw Playhead scrubber line
    if (currentTime >= 0 && currentTime <= duration) {
      const playheadX = (currentTime / duration) * targetWidth;
      ctx.strokeStyle = '#ef4444'; // Red Playhead
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, targetHeight);
      ctx.stroke();

      // Playhead handle
      ctx.fillStyle = '#ef4444';
      ctx.beginPath();
      ctx.arc(playheadX, 5, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [filteredDetections, selectedDetection, hoveredDetection, currentTime, duration, minFreq, maxFreq, viewHeight, zoomLevel]);

  // Synchronize overlay drawing on state updates and window resize
  useEffect(() => {
    renderOverlay();
  }, [renderOverlay]);

  useEffect(() => {
    const handleResize = () => renderOverlay();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [renderOverlay]);

  // Handle Mouse Events (Hover Detection & Click Seek)
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;

    const rect = overlay.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const time = (mouseX / overlay.width) * duration;
    const freq = minFreq + (1 - mouseY / overlay.height) * (maxFreq - minFreq);

    setMousePos({ x: mouseX, y: mouseY, time, freq });

    // Check hit test for filtered detections
    let hit: Detection | null = null;
    for (const det of filteredDetections) {
      const x0 = (det.start_time / duration) * overlay.width;
      const x1 = (det.end_time / duration) * overlay.width;
      const y0 = overlay.height - ((det.max_freq - minFreq) / (maxFreq - minFreq)) * overlay.height;
      const y1 = overlay.height - ((det.min_freq - minFreq) / (maxFreq - minFreq)) * overlay.height;

      if (mouseX >= x0 - 6 && mouseX <= x1 + 6 && mouseY >= y0 - 6 && mouseY <= y1 + 6) {
        hit = det;
        break;
      }
    }
    setHoveredDetection(hit);
  };

  const handleMouseLeave = () => {
    setMousePos(null);
    setHoveredDetection(null);
  };

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;

    const rect = overlay.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const clickTime = (mouseX / overlay.width) * duration;

    if (hoveredDetection) {
      onSelectDetection(hoveredDetection);
      onSeek(hoveredDetection.start_time);
    } else {
      onSeek(clickTime);
      onSelectDetection(null);
    }
  };

  return (
    <div className="flex flex-col gap-2 rounded-2xl bg-[#18181b] border border-[#27272a] p-4 sm:p-5 shadow-2xl">
      {/* Top Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3.5 border-b border-[#27272a]">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-zinc-200">
              超音波頻譜圖 <span className="text-xs font-mono text-emerald-400 font-medium">({(minFreq / 1000).toFixed(0)}kHz ~ {(maxFreq / 1000).toFixed(0)}kHz)</span>
            </h3>
          </div>
          <span className="text-xs px-2.5 py-0.5 rounded-lg bg-[#27272a] border border-[#3f3f46] text-zinc-300 font-mono">
            {filteredDetections.length} 次定位叫聲 <span className="text-[10px] text-zinc-400">(&ge;{Math.round(threshold * 100)}%)</span>
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          {/* Real-time Threshold Slider on Top Toolbar */}
          {onThresholdChange && (
            <div className="flex items-center gap-2 bg-[#27272a] px-2.5 py-1 rounded-xl border border-[#3f3f46]">
              <SlidersHorizontal className="w-3.5 h-3.5 text-zinc-400" />
              <span className="text-zinc-400 text-[11px]">門檻:</span>
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={threshold}
                onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                className="w-16 h-1 bg-[#18181b] rounded-lg appearance-none cursor-pointer accent-emerald-400"
              />
              <span className="font-mono text-emerald-400 text-[11px] font-semibold w-7">
                {Math.round(threshold * 100)}%
              </span>
            </div>
          )}

          {/* Colormap selection */}
          <div className="flex items-center gap-1 bg-[#27272a] px-2 py-1 rounded-xl border border-[#3f3f46]">
            <span className="text-zinc-400 mr-1 text-[11px]">色階:</span>
            {(['magma', 'inferno', 'viridis'] as const).map((cm) => (
              <button
                key={cm}
                onClick={() => setColormap(cm)}
                className={`px-2 py-0.5 rounded-lg uppercase text-[10px] font-mono transition-all cursor-pointer ${
                  colormap === cm
                    ? 'bg-[#18181b] text-emerald-400 font-bold border border-emerald-500/40 shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {cm}
              </button>
            ))}
          </div>

          {/* Height Selector */}
          <div className="flex items-center gap-1 bg-[#27272a] px-2 py-1 rounded-xl border border-[#3f3f46]">
            <Maximize2 className="w-3.5 h-3.5 text-zinc-400 mr-0.5" />
            <span className="text-zinc-400 mr-0.5 text-[11px]">高度:</span>
            {[380, 480, 600].map((h) => (
              <button
                key={h}
                onClick={() => setViewHeight(h)}
                className={`px-1.5 py-0.5 rounded-lg text-[10px] font-mono transition-all cursor-pointer ${
                  viewHeight === h
                    ? 'bg-[#18181b] text-emerald-400 font-bold border border-emerald-500/40 shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {h}px
              </button>
            ))}
          </div>

          {/* Zoom controls */}
          <div className="flex items-center gap-1 bg-[#27272a] px-1.5 py-1 rounded-xl border border-[#3f3f46]">
            <button
              onClick={() => setZoomLevel((z) => Math.max(1, z - 0.5))}
              className="p-1 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-[#18181b] transition-colors cursor-pointer"
              title="縮小"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="px-1.5 font-mono text-[11px] text-zinc-300 font-medium">{zoomLevel}x</span>
            <button
              onClick={() => setZoomLevel((z) => Math.min(5, z + 0.5))}
              className="p-1 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-[#18181b] transition-colors cursor-pointer"
              title="放大"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoomLevel(1)}
              className="p-1 rounded-md text-zinc-400 hover:text-zinc-200 hover:bg-[#18181b] transition-colors cursor-pointer"
              title="重設縮放"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Main Canvas & Frequency Axis */}
      <div className="flex w-full gap-2 items-stretch pt-1">
        {/* Y-Axis (Frequency kHz) */}
        <div
          className="flex flex-col justify-between py-1 text-[11px] font-mono text-zinc-400 select-none text-right w-12 shrink-0"
          style={{ height: `${viewHeight}px` }}
        >
          <span>{(maxFreq / 1000).toFixed(0)}k</span>
          <span>{((maxFreq * 0.75 + minFreq * 0.25) / 1000).toFixed(0)}k</span>
          <span>{((maxFreq * 0.5 + minFreq * 0.5) / 1000).toFixed(0)}k</span>
          <span>{((maxFreq * 0.25 + minFreq * 0.75) / 1000).toFixed(0)}k</span>
          <span>{(minFreq / 1000).toFixed(0)}k</span>
        </div>

        {/* Scrollable Canvas Container */}
        <div
          ref={containerRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          className="relative flex-1 overflow-x-auto overflow-y-hidden rounded-xl border border-[#27272a] bg-[#0c0c0e] cursor-crosshair shadow-inner"
          style={{ height: `${viewHeight}px` }}
        >
          {/* Spectrogram Image Canvas */}
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 h-full pointer-events-none"
            style={{ width: `${100 * zoomLevel}%`, imageRendering: 'auto' }}
          />

          {/* Interaction Overlay Canvas (Boxes, Playhead) */}
          <canvas
            ref={overlayCanvasRef}
            className="absolute top-0 left-0 h-full pointer-events-none"
            style={{ width: `${100 * zoomLevel}%` }}
          />

          {/* Live Hover Tooltip */}
          {mousePos && (
            <div
              className="pointer-events-none absolute z-20 px-2.5 py-1.5 rounded-xl bg-[#18181b]/95 border border-[#3f3f46] text-[11px] font-mono text-zinc-200 shadow-2xl backdrop-blur-md -translate-y-8"
              style={{ left: Math.min(mousePos.x + 10, (containerRef.current?.clientWidth || 500) - 130), top: mousePos.y }}
            >
              <div>時間: <span className="text-zinc-100 font-semibold">{(mousePos.time * 1000).toFixed(1)} ms</span></div>
              <div>頻率: <span className="text-zinc-100 font-semibold">{(mousePos.freq / 1000).toFixed(1)} kHz</span></div>
              {hoveredDetection && (
                <div className="mt-1 pt-1 border-t border-[#27272a] text-emerald-400 font-bold">
                  {hoveredDetection.species} ({(hoveredDetection.confidence * 100).toFixed(1)}%)
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* X-Axis (Time ms/s) */}
      <div className="flex justify-between pl-14 text-[10px] font-mono text-zinc-400 select-none pt-1">
        <span>0 ms</span>
        <span>{(duration * 250).toFixed(0)} ms</span>
        <span>{(duration * 500).toFixed(0)} ms</span>
        <span>{(duration * 750).toFixed(0)} ms</span>
        <span className="text-zinc-300 font-medium">{(duration * 1000).toFixed(0)} ms (總長: {duration.toFixed(2)}s)</span>
      </div>
    </div>
  );

};

/**
 * Colormap RGB generator (Magma, Viridis, Inferno)
 */
function getColormapRGB(v: number, name: 'magma' | 'viridis' | 'inferno'): [number, number, number] {
  if (name === 'inferno') {
    return [
      Math.floor(255 * Math.min(1, v * 1.5)),
      Math.floor(255 * Math.max(0, Math.min(1, (v - 0.2) * 1.6))),
      Math.floor(255 * Math.max(0, Math.min(1, (v - 0.6) * 2.5))),
    ];
  } else if (name === 'viridis') {
    return [
      Math.floor(255 * (0.28 + 0.72 * v * v)),
      Math.floor(255 * (0.1 + 0.9 * v)),
      Math.floor(255 * (0.47 * (1 - v) + 0.1 * v)),
    ];
  } else {
    // Magma
    return [
      Math.floor(255 * Math.min(1, v * 1.3)),
      Math.floor(255 * Math.max(0, Math.min(1, (v - 0.25) * 1.4))),
      Math.floor(255 * Math.max(0, Math.min(1, (v - 0.5) * 2.0))),
    ];
  }
}

