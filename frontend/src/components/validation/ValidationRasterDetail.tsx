import type { ValidationComparisonResult } from "../../api/client";
import { rasterImages, rasterMetricRows } from "./validationComparisonModel";

interface Props {
  result: ValidationComparisonResult;
}

export default function ValidationRasterDetail({ result }: Props) {
  const images = rasterImages(result);
  const rows = rasterMetricRows(result);

  return (
    <section className="control-panel-section">
      <h4>Numeric raster</h4>
      <p>
        <small>
          Display previews may be normalized; comparison metrics use the authoritative numeric
          raster.
        </small>
      </p>
      {!images.absoluteDifference && !images.validPixelChange ? (
        <p>
          <small>No diff images were persisted for this comparison.</small>
        </p>
      ) : (
        <div className="image-grid calibrated-grid">
          {images.absoluteDifference ? (
            <Frame src={images.absoluteDifference} label="Absolute difference heatmap" />
          ) : null}
          {images.validPixelChange ? (
            <Frame src={images.validPixelChange} label="Valid-pixel change map" />
          ) : null}
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
              <th scope="row">Candidate-only valid pixels</th>
              <td>{result.metrics.candidate_only_valid_pixels ?? "—"}</td>
            </tr>
            <tr>
              <th scope="row">Baseline-only valid pixels</th>
              <td>{result.metrics.baseline_only_valid_pixels ?? "—"}</td>
            </tr>
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
      <img src={src} alt={label} style={{ maxWidth: "100%" }} />
      <figcaption>
        <small>{label}</small>
      </figcaption>
    </figure>
  );
}
