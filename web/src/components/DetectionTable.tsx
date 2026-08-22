'use client';

import React, { useState, useMemo } from 'react';
import { AudioAnalysisResult, Detection } from '../lib/types';
import { Download, SlidersHorizontal, BarChart3, ListFilter, Play } from 'lucide-react';

interface DetectionTableProps {
  result: AudioAnalysisResult;
  selectedDetection: Detection | null;
  onSelectDetection: (det: Detection) => void;
  threshold: number;
  onThresholdChange: (threshold: number) => void;
}

// Species Color Badges
const SPECIES_BADGES: { [key: string]: { bg: string; text: string; border: string } } = {
  hiparm: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  myohor: { bg: 'bg-cyan-500/10', text: 'text-cyan-400', border: 'border-cyan-500/30' },
  pipten: { bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/30' },
  tylful: { bg: 'bg-amber-500/10', text: 'text-amber-400', border: 'border-amber-500/30' },
  default: { bg: 'bg-pink-500/10', text: 'text-pink-400', border: 'border-pink-500/30' },
};

export const DetectionTable: React.FC<DetectionTableProps> = ({
  result,
  selectedDetection,
  onSelectDetection,
  threshold,
  onThresholdChange,
}) => {
  const [activeTab, setActiveTab] = useState<'list' | 'stats'>('list');

  // Filter detections by threshold (matching Python detection_score)
  const filteredDetections = useMemo(() => {
    return result.detections.filter((d) => d.detection_score >= threshold);
  }, [result.detections, threshold]);

  // Species breakdown statistics
  const stats = useMemo(() => {
    const counts: { [species: string]: number } = {};
    for (const d of filteredDetections) {
      counts[d.species] = (counts[d.species] || 0) + 1;
    }
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    return sorted;
  }, [filteredDetections]);

  // Export to CSV matching BatDetect2 standard schema
  const handleExportCSV = () => {
    const headers = ['id', 'species', 'start_time', 'end_time', 'duration_ms', 'low_freq_khz', 'high_freq_khz', 'confidence'];
    const rows = filteredDetections.map((d, idx) => [
      idx + 1,
      `"${d.species}"`,
      d.start_time.toFixed(4),
      d.end_time.toFixed(4),
      ((d.end_time - d.start_time) * 1000).toFixed(1),
      (d.min_freq / 1000).toFixed(1),
      (d.max_freq / 1000).toFixed(1),
      d.detection_score.toFixed(3),
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    downloadBlob(csvContent, `${result.filename}_detections.csv`, 'text/csv;charset=utf-8;');
  };

  // Export to JSON
  const handleExportJSON = () => {
    const jsonStr = JSON.stringify(
      {
        filename: result.filename,
        duration_s: result.duration,
        sample_rate_hz: result.sample_rate,
        threshold,
        detections_count: filteredDetections.length,
        detections: filteredDetections,
      },
      null,
      2
    );
    downloadBlob(jsonStr, `${result.filename}_detections.json`, 'application/json');
  };

  const downloadBlob = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-4 rounded-2xl bg-slate-900/90 border border-slate-800 p-5 shadow-xl">
      {/* Top Table Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-800">
        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab('list')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === 'list'
                ? 'bg-slate-800 text-emerald-400 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ListFilter className="w-3.5 h-3.5" />
            叫聲清單 ({filteredDetections.length})
          </button>
          <button
            onClick={() => setActiveTab('stats')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
              activeTab === 'stats'
                ? 'bg-slate-800 text-cyan-400 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            物種統計分佈
          </button>
        </div>

        {/* Filters & Export Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Threshold slider */}
          <div className="flex items-center gap-2 bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400">信心度門檻:</span>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
              className="w-20 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
            />
            <span className="font-mono text-emerald-400 font-semibold w-8">
              {Math.round(threshold * 100)}%
            </span>
          </div>

          {/* Export CSV */}
          <button
            onClick={handleExportCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-emerald-400" />
            CSV
          </button>

          {/* Export JSON */}
          <button
            onClick={handleExportJSON}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            JSON
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {activeTab === 'list' ? (
        <div className="overflow-x-auto rounded-xl border border-slate-800/80">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-950/80 text-[11px] uppercase tracking-wider text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3 px-4 w-12 font-mono">#</th>
                <th className="py-3 px-4 font-semibold">物種標籤</th>
                <th className="py-3 px-4 font-mono">開始時間 (Start)</th>
                <th className="py-3 px-4 font-mono">結束時間 (End)</th>
                <th className="py-3 px-4 font-mono">持續時間 (Dur)</th>
                <th className="py-3 px-4 font-mono">頻率範圍 (kHz)</th>
                <th className="py-3 px-4 font-mono text-right">信心度 (Conf)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50 bg-slate-900/30">
              {filteredDetections.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-8 text-slate-500">
                    在當前門檻 ({Math.round(threshold * 100)}%) 下無偵測叫聲，請調低門檻試試。
                  </td>
                </tr>
              ) : (
                filteredDetections.map((det, idx) => {
                  const isSelected = selectedDetection?.id === det.id;
                  const badge = SPECIES_BADGES[det.species] || SPECIES_BADGES.default;

                  return (
                    <tr
                      key={det.id}
                      onClick={() => onSelectDetection(det)}
                      className={`cursor-pointer transition-colors hover:bg-slate-800/60 ${
                        isSelected ? 'bg-emerald-500/10 border-l-2 border-emerald-400' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-mono text-slate-500">{idx + 1}</td>
                      <td className="py-3 px-4">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded-md font-mono text-[11px] font-semibold border ${badge.bg} ${badge.text} ${badge.border}`}
                        >
                          {det.species}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-300">
                        {det.start_time.toFixed(4)}s ({Math.round(det.start_time * 1000)}ms)
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-300">
                        {det.end_time.toFixed(4)}s ({Math.round(det.end_time * 1000)}ms)
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-400">
                        {((det.end_time - det.start_time) * 1000).toFixed(1)} ms
                      </td>
                      <td className="py-3 px-4 font-mono text-slate-300">
                        {(det.min_freq / 1000).toFixed(1)} - {(det.max_freq / 1000).toFixed(1)} kHz
                      </td>
                      <td className="py-3 px-4 font-mono text-right font-bold text-emerald-400">
                        {(det.detection_score * 100).toFixed(1)}%
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      ) : (
        /* Species Breakdown Statistics Tab */
        <div className="space-y-4 py-2">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {stats.map(([species, count]) => {
              const badge = SPECIES_BADGES[species] || SPECIES_BADGES.default;
              const pct = ((count / filteredDetections.length) * 100).toFixed(1);

              return (
                <div
                  key={species}
                  className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between"
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`px-2 py-0.5 rounded-md font-mono text-xs font-bold border ${badge.bg} ${badge.text} ${badge.border}`}
                    >
                      {species}
                    </span>
                    <span className="text-xl font-bold font-mono text-slate-200">{count}</span>
                  </div>
                  <div className="mt-3">
                    <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                      <span>佔比</span>
                      <span>{pct}%</span>
                    </div>
                    <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-emerald-500 rounded-full"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
