import logoDevai from "../assets/logo_devai.png";
import { PRODUCT_NAV_ITEMS, type ProductArea } from "../productNavigation";

type Props = {
  active: ProductArea;
};

export default function AppHeader({ active }: Props) {
  return (
    <header className="app-header">
      <a className="brand-lockup" href="/operations">
        <img className="devai-logo" src={logoDevai} alt="DevAI" />
        <span className="brand-copy">
          <span>DevAI 3D Acquisition</span>
          <small>Industrial Vision</small>
        </span>
      </a>
      <nav>
        {PRODUCT_NAV_ITEMS.map((item) => (
          <a className={active === item.id || (active === "take" && item.id === "diagnostics") ? "active" : ""} href={item.href} key={item.id}>
            {item.label}
          </a>
        ))}
      </nav>
    </header>
  );
}
