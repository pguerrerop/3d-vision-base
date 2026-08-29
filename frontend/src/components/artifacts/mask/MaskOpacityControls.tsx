type Props = {
  maskOpacity: number;
  referenceOpacity: number;
  maskOpacityDisabled?: boolean;
  referenceOpacityDisabled?: boolean;
  onMaskOpacityChange?: (value: number) => void;
  onReferenceOpacityChange?: (value: number) => void;
};

export default function MaskOpacityControls({
  maskOpacity,
  referenceOpacity,
  maskOpacityDisabled = false,
  referenceOpacityDisabled = false,
  onMaskOpacityChange,
  onReferenceOpacityChange,
}: Props) {
  return (
    <div className="mask-opacity-row">
      <label className="mask-opacity-control">
        <span>Mask opacity</span>
        <input
          aria-label="Mask overlay opacity"
          disabled={maskOpacityDisabled}
          max={0.8}
          min={0.2}
          onChange={(event) => onMaskOpacityChange?.(Number(event.target.value))}
          step={0.01}
          type="range"
          value={maskOpacity}
        />
        <strong>{Math.round(maskOpacity * 100)}%</strong>
      </label>
      <label className="mask-opacity-control">
        <span>Reference opacity</span>
        <input
          aria-label="Mask reference opacity"
          disabled={referenceOpacityDisabled}
          max={1}
          min={0.2}
          onChange={(event) => onReferenceOpacityChange?.(Number(event.target.value))}
          step={0.01}
          type="range"
          value={referenceOpacity}
        />
        <strong>{Math.round(referenceOpacity * 100)}%</strong>
      </label>
    </div>
  );
}
