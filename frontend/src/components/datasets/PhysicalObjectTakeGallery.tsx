import { fileUrl } from "../../api/client";
import { buildStudioDeepLink } from "../../studioDeepLink";

export default function PhysicalObjectTakeGallery({
  takes,
  loading = false,
  datasetId,
  physicalObjectId,
}: {
  takes: Array<{ session_id: string; take_id: string; metadata: Record<string, unknown> }>;
  loading?: boolean;
  datasetId?: string | null;
  physicalObjectId?: string | null;
}) {
  return (
    <section className="entity-section">
      <h4>Observed Takes</h4>
      {loading ? <small>Loading observed takes…</small> : null}
      {!loading && takes.length === 0 ? <small>No takes are currently linked to this physical object.</small> : null}
      <div className="ml-set-sample-row">
        {takes.map((row) => {
          const thumbnailPath = String(row.metadata.thumbnail_path || "").trim();
          const studioHref = buildStudioDeepLink({
            take_id: row.take_id,
            dataset_id: datasetId || null,
            session_id: row.session_id,
            physical_object_id: physicalObjectId || null,
          });
          return (
            <a key={row.take_id} href={studioHref} target="_blank" rel="noreferrer" className="ml-set-sample-card" title={`Open ${row.take_id} in Studio`}>
              {thumbnailPath ? <img src={fileUrl(row.take_id, thumbnailPath)} alt={row.take_id} loading="lazy" /> : <span className="datasets-thumb-placeholder">-</span>}
              <small>{row.take_id}</small>
            </a>
          );
        })}
      </div>
    </section>
  );
}
