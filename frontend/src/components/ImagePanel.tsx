type Props = {
  title: string;
  src?: string | null;
  emptyMessage?: string;
};

export default function ImagePanel({ title, src, emptyMessage = "No image" }: Props) {
  return (
    <section className="image-panel">
      <div className="panel-title">{title}</div>
      {src ? <img src={src} alt={title} /> : <div className="empty-image">{emptyMessage}</div>}
    </section>
  );
}
