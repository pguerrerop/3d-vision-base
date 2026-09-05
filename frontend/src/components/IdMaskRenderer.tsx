import { useEffect, useRef, useState } from "react";
import { colorizeIdMaskImageData } from "./idMaskArtifacts";

type Props = {
  src: string;
  alt?: string;
  className?: string;
  loading?: "lazy" | "eager";
};

// Loads the raw id-mask PNG into an offscreen canvas, recolors its integer label values, and
// displays the recolored canvas instead of the raw (near-black) grayscale image. Falls back to a
// plain <img> if the image can't be read as pixel data (e.g. a cross-origin host without CORS
// headers taints the canvas) so a broken renderer never means a broken artifact.
export default function IdMaskRenderer({ src, alt, className, loading = "lazy" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fallback, setFallback] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setFallback(false);
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setFallback(true);
        return;
      }
      ctx.drawImage(image, 0, 0);
      try {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        colorizeIdMaskImageData(imageData);
        ctx.putImageData(imageData, 0, 0);
      } catch {
        setFallback(true);
      }
    };
    image.onerror = () => {
      if (!cancelled) setFallback(true);
    };
    image.src = src;
    return () => {
      cancelled = true;
    };
  }, [src]);

  if (fallback) {
    return <img src={src} alt={alt ?? ""} className={className} loading={loading} />;
  }
  return <canvas ref={canvasRef} className={className} role="img" aria-label={alt ?? "Colorized id mask"} />;
}
