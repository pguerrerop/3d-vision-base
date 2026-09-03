import type { ValidationComparisonResult } from "../../api/client";
import { maskImages, maskMetricRows } from "./validationComparisonModel";

interface Props {
  result: ValidationComparisonResult;
}

export default function ValidationMaskDetail({ result }: Props) {
  const images = maskImages(result);
  const rows = maskMetricRows(result);

  return (
    <section className="control-panel-section">
      <h4>Binary mask</h4>
      {!Object.values(images).some(Boolean) ? (
        <p>
          <small>No diff images were persisted for this comparison.</small>
        </p>
      ) : (
        <div className="image-grid">
          {images.baseline ? <Frame src={images.baseline} label="Baseline" /> : null}
          {images.candidate ? <Frame src={images.candidate} label="Candidate" /> : null}
          {images.combined ? <Frame src={images.combined} label="Combined diff" /> : null}
          {images.added ? <Frame src={images.added} label="Added only" /> : null}
          {images.removed ? <Frame src={images.removed} label="Removed only" /> : null}
        </div>
      )}
      <div className="table-wrap">
        <table>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                <td>{row.value}</td>
              </tr>
            ))}
            <tr>
              <th scope="row">Thresholds</th>
              <td>
                <code>{JSON.stringify(result.thresholds)}</code>
              </td>
            </tr>
            {result.reasons.length ? (
              <tr>
                <th scope="row">Failure reasons</th>
                <td>{result.reasons.join(", ")}</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Frame({ src, label }: { src: string; label: string }) {
  return (
    <figure style={{ margin: 0 }}>
      <img src={src} alt={label} style={{ maxWidth: "100%", imageRendering: "pixelated" }} />
      <figcaption>
        <small>{label}</small>
      </figcaption>
    </figure>
  );
}
