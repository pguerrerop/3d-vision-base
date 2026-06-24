import { fileUrl } from "../../api/client";

export default function PhysicalObjectTakeGallery({
  takes,
}: {
  takes: Array<{ session_id: string; take_id: string; metadata: Record<string, unknown> }>;
}) {
  return (
    <section className="entity-section">
      <h4>Observed Takes</h4>
      <div className="ml-set-sample-row">
        {takes.map((row) => {
          const thumbnailPath = String(row.metadata.thumbnail_path || "").trim();
          return (
            <a key={row.take_id} href={`/takes/${encodeURIComponent(row.take_id)}`} className="ml-set-sample-card" title={`${row.take_id} · ${row.session_id}`}>
              {thumbnailPath ? <img src={fileUrl(row.take_id, thumbnailPath)} alt={row.take_id} loading="lazy" /> : <span className="datasets-thumb-placeholder">-</span>}
              <small>{row.take_id}</small>
            </a>
          );
        })}
      </div>
    </section>
  );
}
