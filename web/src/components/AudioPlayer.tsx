'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Volume2, FastForward, Gauge } from 'lucide-react';

interface AudioPlayerProps {
  audioBuffer: ArrayBuffer | null;
  sampleRate: number;
  duration: number;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  audioBuffer,
  sampleRate,
  duration,
  currentTime,
  onTimeUpdate,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState<number>(0.1); // Default 0.1x (10x time expansion for ultrasound)
  const [volume, setVolume] = useState<number>(0.8);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const decodedAudioBufferRef = useRef<AudioBuffer | null>(null);
  const startOffsetRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const animFrameRef = useRef<number | null>(null);

  // Initialize Web Audio Context & Decode
  useEffect(() => {
    if (!audioBuffer) return;

    const ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    audioCtxRef.current = ctx;

    const gainNode = ctx.createGain();
    gainNode.gain.value = volume;
    gainNode.connect(ctx.destination);
    gainNodeRef.current = gainNode;

    // Decode WAV audio for playback (standard Web Audio decode or fallback)
    ctx.decodeAudioData(audioBuffer.slice(0)).then((buf) => {
      decodedAudioBufferRef.current = buf;
    }).catch((err) => {
      console.warn('AudioContext decodeAudioData error, creating raw buffer:', err);
    });

    return () => {
      stopPlayback();
      ctx.close();
    };
  }, [audioBuffer]);

  // Adjust volume
  useEffect(() => {
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = volume;
    }
  }, [volume]);

  // Stop playback helper
  const stopPlayback = () => {
    if (sourceNodeRef.current) {
      try {
        sourceNodeRef.current.stop();
        sourceNodeRef.current.disconnect();
      } catch {
        // already stopped
      }
      sourceNodeRef.current = null;
    }
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setIsPlaying(false);
  };

  // Start playback from current offset
  const startPlayback = (offsetSeconds: number) => {
    if (!audioCtxRef.current || !decodedAudioBufferRef.current) return;

    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume();
    }

    stopPlayback();

    const ctx = audioCtxRef.current;
    const source = ctx.createBufferSource();
    source.buffer = decodedAudioBufferRef.current;
    source.playbackRate.value = playbackRate;
    source.connect(gainNodeRef.current!);

    const validOffset = Math.max(0, Math.min(offsetSeconds, duration));
    startOffsetRef.current = validOffset;
    startTimeRef.current = ctx.currentTime;

    source.start(0, validOffset);
    sourceNodeRef.current = source;
    setIsPlaying(true);

    source.onended = () => {
      if (sourceNodeRef.current === source) {
        setIsPlaying(false);
        onTimeUpdate(0);
      }
    };

    // Update playhead progress via requestAnimationFrame
    const updateProgress = () => {
      if (!sourceNodeRef.current) return;
      const elapsed = (ctx.currentTime - startTimeRef.current) * playbackRate;
      const current = startOffsetRef.current + elapsed;

      if (current >= duration) {
        setIsPlaying(false);
        onTimeUpdate(duration);
        return;
      }

      onTimeUpdate(current);
      animFrameRef.current = requestAnimationFrame(updateProgress);
    };
    animFrameRef.current = requestAnimationFrame(updateProgress);
  };

  const handlePlayPause = () => {
    if (isPlaying) {
      stopPlayback();
    } else {
      const startAt = currentTime >= duration - 0.01 ? 0 : currentTime;
      startPlayback(startAt);
    }
  };

  const handleRateChange = (rate: number) => {
    setPlaybackRate(rate);
    if (isPlaying) {
      // Restart at current time with new playback rate
      startPlayback(currentTime);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const seekTime = parseFloat(e.target.value);
    onTimeUpdate(seekTime);
    if (isPlaying) {
      startPlayback(seekTime);
    }
  };

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-2xl bg-slate-900/90 border border-slate-800 shadow-xl">
      {/* Play / Pause / Reset */}
      <div className="flex items-center gap-2">
        <button
          onClick={handlePlayPause}
          className="flex items-center justify-center w-11 h-11 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold shadow-lg shadow-emerald-500/20 transition-all hover:scale-105 active:scale-95"
          title={isPlaying ? '暫停' : '降速播放'}
        >
          {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-0.5" />}
        </button>

        <button
          onClick={() => {
            stopPlayback();
            onTimeUpdate(0);
          }}
          className="flex items-center justify-center w-9 h-9 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
          title="重設播放位置"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Progress Bar */}
      <div className="flex-1 w-full flex items-center gap-3">
        <span className="text-xs font-mono text-slate-400 w-14 text-right">
          {(currentTime * 1000).toFixed(0)} ms
        </span>
        <input
          type="range"
          min="0"
          max={duration || 1}
          step="0.001"
          value={currentTime}
          onChange={handleSeek}
          className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
        />
        <span className="text-xs font-mono text-slate-400 w-14">
          {(duration * 1000).toFixed(0)} ms
        </span>
      </div>

      {/* Ultrasound Time Expansion Speed Control */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1 bg-slate-950 px-2 py-1 rounded-lg border border-slate-800 text-xs">
          <Gauge className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-500 text-[11px] mr-1">降速聆聽:</span>
          {[0.05, 0.1, 0.2, 0.5, 1.0].map((rate) => (
            <button
              key={rate}
              onClick={() => handleRateChange(rate)}
              className={`px-1.5 py-0.5 rounded font-mono text-[11px] transition-colors ${
                playbackRate === rate
                  ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/40'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {rate === 1.0 ? '1x' : `${rate}x`}
            </button>
          ))}
        </div>

        {/* Volume */}
        <div className="hidden md:flex items-center gap-1.5 text-slate-400">
          <Volume2 className="w-4 h-4 text-slate-400" />
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={volume}
            onChange={(e) => setVolume(parseFloat(e.target.value))}
            className="w-16 h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-400"
          />
        </div>
      </div>
    </div>
  );
};
