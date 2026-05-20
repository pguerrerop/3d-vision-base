import type { StudioObject } from "./studioWorkspaceModel";

type Props = {
  objects: StudioObject[];
  selectedObjectId: number | null;
  hoveredObjectId?: number | null;
  onSelect: (objectId: number) => void;
  onHover?: (objectId: number | null) => void;
};

export default function StudioObjectList({ objects, selectedObjectId, hoveredObjectId = null, onSelect, onHover }: Props) {
  if (!objects.length) {
    return <div className="empty-state">No candidate objects are available for this take.</div>;
  }
  return (
    <div className="studio-object-list">
      {objects.map((object) => (
        <button
          className={object.object_id === selectedObjectId ? "active" : object.object_id === hoveredObjectId ? "hover" : ""}
          key={object.object_id}
          onClick={() => onSelect(object.object_id)}
          onMouseEnter={() => onHover?.(object.object_id)}
          onMouseLeave={() => onHover?.(null)}
          type="button"
        >
          <strong>Object #{object.object_id}</strong>
          <span>{object.class_name}</span>
          <small>{object.confidence == null ? "confidence unavailable" : `${Math.round(object.confidence * 100)}% confidence`}</small>
          <small>{object.filter_status ?? "candidate"}</small>
        </button>
      ))}
    </div>
  );
}
