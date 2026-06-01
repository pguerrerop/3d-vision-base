export default function EntityExportActions({ onExport }: { onExport: () => void }) {
  return (
    <div className="entity-export-actions">
      <button type="button" onClick={onExport}>Export session JSON</button>
    </div>
  );
}
