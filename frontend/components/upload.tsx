"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  Camera,
  ImagePlus,
  Loader2,
  ScanSearch,
  UploadCloud,
  X,
  Activity,
  ChevronRight,
  CheckCircle2,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { GlassCard, GhostButton, PrimaryButton } from "@/components/ui";
import { predictImage } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";
import { toast } from "react-hot-toast";

type Props = {
  onResult: (result: PredictResponse, previewUrl: string | null) => void;
};

const pipelineSteps = [
  { key: "upload", label: "Upload", icon: UploadCloud },
  { key: "process", label: "Process", icon: Activity },
  { key: "analyze", label: "Analyze", icon: ScanSearch },
  { key: "complete", label: "Complete", icon: CheckCircle2 },
];

export function UploadAnalyze({ onResult }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [scanPhase, setScanPhase] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [streamActive, setStreamActive] = useState(false);
  const [captured, setCaptured] = useState(false);
  const [statusText, setStatusText] = useState<string>("No file selected");
  const [errorText, setErrorText] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!cameraOpen) {
      stopCamera();
      return;
    }
    let active = true;
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then((mediaStream) => {
        if (!active) {
          mediaStream.getTracks().forEach((track) => track.stop());
          return;
        }
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          videoRef.current.play().catch(() => {});
          setStreamActive(true);
          setCaptured(false);
        }
      })
      .catch(async () => {
        if (!active) return;
        try {
          const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          if (!active) {
            mediaStream.getTracks().forEach((track) => track.stop());
            return;
          }
          if (videoRef.current) {
            videoRef.current.srcObject = mediaStream;
            videoRef.current.play().catch(() => {});
            setStreamActive(true);
            setCaptured(false);
          }
        } catch {
          if (!active) return;
          toast.error("Camera access is unavailable in this browser.");
          setCameraOpen(false);
        }
      });
    return () => {
      active = false;
      stopCamera();
    };
  }, [cameraOpen]);

  useEffect(() => {
    if (analyzing) {
      const interval = setInterval(() => {
        setScanPhase((p) => (p >= 3 ? 3 : p + 1));
      }, 700);
      return () => clearInterval(interval);
    } else {
      setScanPhase(0);
    }
  }, [analyzing]);

  function stopCamera() {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((track) => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setStreamActive(false);
  }

  function handleFile(nextFile: File | null) {
    setErrorText(null);
    if (!nextFile) {
      setStatusText("No file selected");
      return;
    }
    if (!nextFile.type.startsWith("image/")) {
      setErrorText("Please select a valid image file.");
      setStatusText("Invalid file type");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setStatusText(`Selected: ${nextFile.name} (${(nextFile.size / 1024).toFixed(1)} KB)`);
    setZoom(1);
    setCaptured(false);
  }

  function clearSelection() {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setCaptured(false);
  }

  async function analyze() {
    setErrorText(null);
    if (!file) {
      setErrorText("Choose an image first.");
      setStatusText("No file selected");
      return;
    }
    setAnalyzing(true);
    setStatusText(`Analyzing: ${file.name}...`);
    try {
      const result = await predictImage(file);
      setStatusText("Analysis completed.");
      onResult(result, previewUrl);
      toast.success("Analysis completed.");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Analysis failed.";
      setErrorText(message);
      setStatusText(`Failed: ${message}`);
      toast.error(message);
    } finally {
      setAnalyzing(false);
    }
  }

  async function captureFrame() {
    setErrorText(null);
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) {
      setErrorText("Camera is not ready.");
      return;
    }
    if (!streamActive) {
      setErrorText("Camera stream is not active.");
      return;
    }
    try {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const context = canvas.getContext("2d");
      if (!context) {
        setErrorText("Unable to capture frame.");
        return;
      }
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
      if (!blob) {
        setErrorText("Failed to create image from camera.");
        return;
      }
      const capturedFile = new File([blob], `capture-${Date.now()}.png`, { type: "image/png" });
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setFile(capturedFile);
      setPreviewUrl(URL.createObjectURL(capturedFile));
      setStatusText(`Captured frame: ${capturedFile.name}`);
      setCaptured(true);
      setCameraOpen(false);
      stopCamera();
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Camera capture failed.");
    }
  }

  return (
    <GlassCard className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Upload & analyze</p>
          <h3 className="mt-2 font-heading text-2xl text-slate-900 dark:text-white">Start from an image or your webcam</h3>
        </div>
        <GhostButton
          type="button"
          onClick={() => {
            if (!cameraOpen) setCaptured(false);
            setCameraOpen((value) => !value);
          }}
        >
          <Camera className="mr-2 h-4 w-4" />
          {cameraOpen ? "Close camera" : "Open camera"}
        </GhostButton>
      </div>

      <div className="space-y-2 rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.03] p-4 text-xs text-slate-600 dark:text-slate-300">
        <div className="flex items-center justify-between gap-3">
          <span className="text-slate-500 dark:text-slate-400">Status</span>
          <span className="font-semibold text-slate-900 dark:text-white">{statusText}</span>
        </div>
        {errorText && (
          <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 p-2 text-rose-200">{errorText}</div>
        )}
      </div>

      {(file || (cameraOpen && !captured)) ? (
        <div className="grid gap-5 lg:grid-cols-[1.1fr,0.9fr]">
          {!file ? (
            <div className="space-y-4">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center justify-center rounded-full border border-slate-200 dark:border-white/12 bg-slate-100 dark:bg-white/5 px-5 py-3 text-sm font-semibold text-slate-900 dark:text-white transition hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-200 dark:hover:bg-white/[0.08]"
              >
                <ImagePlus className="mr-2 h-4 w-4" />
                Browse Image
              </button>
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragActive(false);
                  handleFile(event.dataTransfer.files[0] ?? null);
                }}
                className={`
                  relative overflow-hidden rounded-3xl border-2 border-dashed p-8 transition-all duration-300 cursor-pointer
                   ${dragActive ? "border-accent-300 bg-accent-400/10 scale-[1.01]" : "border-slate-200 dark:border-white/12 bg-slate-100 dark:bg-white/[0.02] hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-200 dark:hover:bg-white/[0.04]"}
                `}
              >
                <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(24,189,224,0.06),transparent_70%)] opacity-0 transition-opacity duration-300" style={{ opacity: dragActive ? 1 : 0 }} />
                <div className="relative flex cursor-pointer flex-col items-center justify-center gap-4 text-center">
                  <motion.div
                    animate={{ y: dragActive ? -4 : 0, scale: dragActive ? 1.05 : 1 }}
                    transition={{ type: "spring", stiffness: 400, damping: 20 }}
                    className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 dark:bg-white/[0.06] text-accent-300 ring-1 ring-slate-200 dark:ring-white/10"
                  >
                    <UploadCloud size={28} />
                  </motion.div>
                  <div>
                    <p className="text-sm font-semibold text-slate-900 dark:text-white">Drop an image here or browse</p>
                    <p className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">PNG, JPEG, or WebP. Minimum 32 px.</p>
                  </div>
                  <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} />
                </div>
              </div>
            </div>
          ) : (
            <GlassCard className="relative overflow-hidden p-0">
              <div className="absolute right-4 top-4 z-10 flex gap-2">
                <button
                  type="button"
                  onClick={clearSelection}
                  disabled={analyzing}
                  className="rounded-full bg-black/40 p-2 text-white backdrop-blur-md transition hover:bg-black/60 disabled:opacity-50"
                >
                  <X size={16} />
                </button>
              </div>

              <div className="relative aspect-[4/3] overflow-hidden bg-black/30">
                <motion.img
                  src={previewUrl ?? ""}
                  alt="Selected skin image preview"
                  className="h-full w-full object-cover"
                  style={{ transform: `scale(${zoom})`, transformOrigin: "center" }}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4 }}
                />

                {analyzing && (
                  <div className="absolute inset-0 bg-slate-900/40 dark:bg-slate-950/40 backdrop-blur-[2px]">
                    <div className="absolute inset-0 overflow-hidden">
                      <motion.div
                        className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-accent-400 to-transparent shadow-[0_0_20px_rgba(24,189,224,0.8)]"
                        animate={{ top: ["0%", "100%", "0%"] }}
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      />
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="flex items-center gap-3 rounded-full bg-black/60 px-6 py-3 backdrop-blur-md">
                        <Loader2 className="h-5 w-5 animate-spin text-accent-300" />
                        <span className="text-sm font-semibold text-white">Analyzing...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-4 p-5">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>Zoom / crop</span>
                  <span>{Math.round(zoom * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="1.8"
                  step="0.01"
                  value={zoom}
                  onChange={(event) => setZoom(Number(event.target.value))}
                  className="w-full accent-cyan-300 h-1.5 rounded-full appearance-none bg-white/[0.08] cursor-pointer"
                  disabled={analyzing}
                />

                <div className="flex flex-wrap gap-3">
                  <PrimaryButton type="button" onClick={analyze} disabled={analyzing}>
                    {analyzing ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <ScanSearch className="mr-2 h-4 w-4" />
                    )}
                    {analyzing ? "Analyzing..." : "Analyze image"}
                  </PrimaryButton>
                  <GhostButton type="button" onClick={clearSelection} disabled={analyzing}>
                    Clear image
                  </GhostButton>
                </div>
              </div>
            </GlassCard>
          )}
          {cameraOpen && !captured ? (
            <GlassCard className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-slate-900 dark:text-white">Webcam preview</p>
                <span className={`flex items-center gap-1.5 text-xs ${streamActive ? "text-emerald-300" : "text-slate-500 dark:text-slate-400"}`}>
                  <span className={`relative rounded-full h-2 w-2 ${streamActive ? "bg-emerald-400" : "bg-slate-400"}`} />
                </span>
                {streamActive ? "Live" : "Waiting"}
              </div>
              <div className="relative overflow-hidden rounded-2xl border border-slate-200 dark:border-white/10 bg-black/30">
                <video ref={videoRef} autoPlay playsInline muted className="aspect-video w-full object-cover" />
                {!streamActive && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="h-8 w-8 animate-spin rounded-full border-2 border-white/20 border-t-accent-400" />
                  </div>
                )}
              </div>
              <canvas ref={canvasRef} className="hidden" />
              <div className="flex flex-wrap gap-3">
                <PrimaryButton type="button" onClick={captureFrame} disabled={!streamActive}>
                  <ImagePlus className="mr-2 h-4 w-4" /> Capture frame
                </PrimaryButton>
                <GhostButton type="button" onClick={() => { setCameraOpen(false); setCaptured(false); }}>
                  Stop camera
                </GhostButton>
              </div>
            </GlassCard>
          ) : file ? (
              <GlassCard className="flex flex-col items-center justify-center text-center">
                <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 dark:bg-white/[0.05] ring-1 ring-slate-200 dark:ring-white/10">
                  <Camera className="text-accent-300" size={20} />
                </div>
                <p className="text-sm font-semibold text-slate-900 dark:text-white">Webcam Ready</p>
                <p className="mt-2 text-xs leading-6 text-slate-500 dark:text-slate-400">
                  Open the camera to capture a live photo of the skin area for analysis.
                </p>
              </GlassCard>
          ) : null}
        </div>
      ) : (
        <div className="space-y-4">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center justify-center rounded-full border border-slate-200 dark:border-white/12 bg-slate-100 dark:bg-white/5 px-5 py-3 text-sm font-semibold text-slate-900 dark:text-white transition hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-200 dark:hover:bg-white/[0.08]"
          >
            <ImagePlus className="mr-2 h-4 w-4" />
            Browse Image
          </button>
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              handleFile(event.dataTransfer.files[0] ?? null);
            }}
            className={`
              relative overflow-hidden rounded-3xl border-2 border-dashed p-8 transition-all duration-300 cursor-pointer
              ${dragActive ? "border-accent-300 bg-accent-400/10 scale-[1.01]" : "border-slate-200 dark:border-white/12 bg-slate-100 dark:bg-white/[0.02] hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-200 dark:hover:bg-white/[0.04]"}
            `}
          >
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(24,189,224,0.06),transparent_70%)] opacity-0 transition-opacity duration-300" style={{ opacity: dragActive ? 1 : 0 }} />
            <div className="relative flex cursor-pointer flex-col items-center justify-center gap-4 text-center">
              <motion.div
                animate={{ y: dragActive ? -4 : 0, scale: dragActive ? 1.05 : 1 }}
                transition={{ type: "spring", stiffness: 400, damping: 20 }}
                className="flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 dark:bg-white/[0.06] text-accent-300 ring-1 ring-slate-200 dark:ring-white/10"
              >
                <UploadCloud size={28} />
              </motion.div>
              <div>
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">Drop an image here or browse</p>
                  <p className="mt-2 text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">PNG, JPEG, or WebP. Minimum 32 px.</p>
              </div>
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={(event) => handleFile(event.target.files?.[0] ?? null)} />
            </div>
          </div>
        </div>
      )}

      {!file && !analyzing && (
        <div className="rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.03] p-5">
          <p className="text-sm font-semibold text-slate-900 dark:text-white">How it works</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-4">
            {pipelineSteps.map((step, index) => (
              <div key={step.key} className="flex items-center gap-3">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-100 dark:bg-white/[0.06] text-accent-300 ring-1 ring-slate-200 dark:ring-white/10">
                  <step.icon size={14} />
                </div>
                <div className="text-left">
                  <p className="text-xs font-semibold text-slate-900 dark:text-white">{step.label}</p>
                  <p className="text-[10px] text-slate-500 dark:text-slate-400">Step {index + 1}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {analyzing && (
        <div className="rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.03] p-5">
          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-3">
            <span>Analysis progress</span>
            <span>{Math.round((scanPhase / 3) * 100)}%</span>
          </div>
          <div className="flex items-center gap-2">
            {pipelineSteps.slice(0, 4).map((step, index) => (
              <div key={step.key} className="flex items-center gap-2 flex-1">
                <div className={`flex h-6 w-6 items-center justify-center rounded-lg transition-all duration-300 ${scanPhase >= index ? "bg-accent-400/20 text-accent-300 ring-1 ring-accent-300/30" : "bg-slate-100 dark:bg-white/[0.04] text-slate-500 dark:text-slate-400"}`}>
                  {scanPhase > index ? <CheckCircle2 size={12} /> : <step.icon size={12} />}
                </div>
                {index < 3 && <div className={`h-px flex-1 transition-colors duration-300 ${scanPhase > index ? "bg-accent-400/30" : "bg-slate-200 dark:bg-white/[0.06]"}`} />}
              </div>
            ))}
          </div>
        </div>
      )}
    </GlassCard>
  );
}
