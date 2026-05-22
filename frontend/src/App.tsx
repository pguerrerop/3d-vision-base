import OperatorPage from "./pages/OperatorPage";
import DebugPage from "./pages/DebugPage";
import TakeDetailPage from "./pages/TakeDetailPage";
import CalibrationPage from "./pages/CalibrationPage";
import ProcessingLabPage from "./pages/ProcessingLabPage";
import RuntimePage from "./pages/RuntimePage";
import AppHeader from "./components/AppHeader";
import { productAreaForPath } from "./productNavigation";

type Route =
  | { name: "operations" }
  | { name: "diagnostics" }
  | { name: "runtime" }
  | { name: "calibration" }
  | { name: "studio" }
  | { name: "take"; takeId: string };

function currentRoute(): Route {
  const path = window.location.pathname;
  if (path.startsWith("/takes/")) {
    return { name: "take", takeId: decodeURIComponent(path.replace("/takes/", "")) };
  }
  const area = productAreaForPath(path);
  return { name: area === "take" ? "operations" : area };
}

export default function App() {
  const route = currentRoute();
  return (
    <>
      <AppHeader active={route.name} />
      {route.name === "calibration" && <CalibrationPage />}
      {route.name === "studio" && <ProcessingLabPage />}
      {route.name === "runtime" && <RuntimePage />}
      {route.name === "diagnostics" && <DebugPage />}
      {route.name === "take" && <TakeDetailPage takeId={route.takeId} />}
      {route.name === "operations" && <OperatorPage />}
    </>
  );
}
