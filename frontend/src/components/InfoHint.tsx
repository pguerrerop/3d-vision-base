import { useEffect, useId, useState, type ReactNode } from "react";
import type { StudioHelpEntry } from "./studioHelp";

type Props = {
  content: StudioHelpEntry | null;
  className?: string;
  label?: string;
  defaultOpen?: boolean;
};

function DetailRow(props: { label: string; value: ReactNode }) {
  return (
    <div className="info-hint-detail-row">
      <strong>{props.label}</strong>
      <p>{props.value}</p>
    </div>
  );
}

export default function InfoHint({ content, className = "", label = "Open help", defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const titleId = useId();

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  if (!content) return null;

  return (
    <>
      <button
        type="button"
        aria-label={`${label}: ${content.title}`}
        aria-haspopup="dialog"
        className={`info-hint-button ${className}`.trim()}
        onClick={(event) => {
          event.stopPropagation();
          setOpen(true);
        }}
      >
        i
      </button>
      {open ? (
        <div
          className="modal-backdrop info-hint-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
          onClick={() => setOpen(false)}
        >
          <div className="modal-panel info-hint-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h3 id={titleId}>{content.title}</h3>
              <button type="button" onClick={() => setOpen(false)}>Close</button>
            </div>
            <div className="info-hint-body">
              <p className="info-hint-summary">{content.summary}</p>
              {content.details.split(/\n\n+/).filter(Boolean).map((paragraph) => (
                <p key={paragraph.slice(0, 48)}>{paragraph}</p>
              ))}
              {content.consumes ? <DetailRow label="Consumes" value={content.consumes} /> : null}
              {content.produces ? <DetailRow label="Produces" value={content.produces} /> : null}
              {content.supportType ? <DetailRow label="Support impact" value={content.supportType} /> : null}
              {content.fragmentedGuidance ? <DetailRow label="If belt is fragmented" value={content.fragmentedGuidance} /> : null}
              {content.mergedGuidance ? <DetailRow label="If object and belt are merged" value={content.mergedGuidance} /> : null}
              {content.overTuningRisk ? <DetailRow label="Over-tuning risk" value={content.overTuningRisk} /> : null}
              {content.whenToIncrease ? <DetailRow label="When to increase" value={content.whenToIncrease} /> : null}
              {content.whenToDecrease ? <DetailRow label="When to decrease" value={content.whenToDecrease} /> : null}
              {content.optionDetails?.length ? (
                <div className="info-hint-detail-row">
                  <strong>Options</strong>
                  <div>
                    {content.optionDetails.map((option) => (
                      <p key={option.value}>
                        <strong>{option.label}</strong>: {option.description}
                      </p>
                    ))}
                  </div>
                </div>
              ) : null}
              {content.affects?.length ? <DetailRow label="Affects" value={content.affects.join(" • ")} /> : null}
              {content.symptoms?.length ? <DetailRow label="Typical symptoms" value={content.symptoms.join(" • ")} /> : null}
              {content.related?.length ? <DetailRow label="Related artifacts" value={content.related.join(" • ")} /> : null}
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
