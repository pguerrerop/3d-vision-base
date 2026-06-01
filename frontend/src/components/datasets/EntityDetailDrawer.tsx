import { useEffect, type ReactNode } from "react";

type Props = {
  open: boolean;
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  mode?: "compact" | "wide";
  headerExtras?: ReactNode;
};

export default function EntityDetailDrawer({ open, title, subtitle, onClose, children, footer, mode = "compact", headerExtras }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <aside className={`entity-drawer ${mode === "wide" ? "wide" : "compact"}`} aria-label={`${title} details`}>
      <header className="entity-drawer-header">
        <div>
          <h3>{title}</h3>
          {subtitle ? <small>{subtitle}</small> : null}
          {headerExtras ? <div className="entity-drawer-header-extras">{headerExtras}</div> : null}
        </div>
        <button type="button" onClick={onClose} aria-label="Close details">Close</button>
      </header>
      <div className="entity-drawer-body">{children}</div>
      {footer ? <footer className="entity-drawer-footer">{footer}</footer> : null}
    </aside>
  );
}
