import { type ReactNode } from "react";

export default function WizardStepLayout({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className="concept-panel compact">
      <h3>{title}</h3>
      <small>{description}</small>
      <div style={{ marginTop: 8 }}>{children}</div>
    </section>
  );
}
