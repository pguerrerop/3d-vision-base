import type { MLSetSummaryResponse } from "../../api/client";

export default function MLSetTaskDefinitions({ tasks }: { tasks: MLSetSummaryResponse["derived_tasks"] }) {
  return (
    <section className="entity-section">
      <h4>Derived Tasks</h4>
      <div className="ml-set-task-list">
        {tasks.map((task) => (
          <article key={task.task_id} className="ml-set-task-card">
            <strong>{task.task_id}</strong>
            <small>samples: {task.effective_sample_count}</small>
            <small>included: {task.included_classes.join(", ") || "-"}</small>
            <small>excluded: {task.excluded_classes.join(", ") || "-"}</small>
          </article>
        ))}
      </div>
    </section>
  );
}
