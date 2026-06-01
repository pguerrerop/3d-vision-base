export type SummaryCard = { label: string; value: string | number };

export default function EntitySummaryCards({ cards }: { cards: SummaryCard[] }) {
  return (
    <div className="entity-summary-cards">
      {cards.map((card) => (
        <article key={card.label}>
          <small>{card.label}</small>
          <strong>{card.value}</strong>
        </article>
      ))}
    </div>
  );
}
