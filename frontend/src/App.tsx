import OperatorPage from "./pages/OperatorPage";
import DebugPage from "./pages/DebugPage";
import TakeDetailPage from "./pages/TakeDetailPage";
import CalibrationPage from "./pages/CalibrationPage";
import ProcessingLabPage from "./pages/ProcessingLabPage";
import PipelineWalkthroughPage from "./pages/PipelineWalkthroughPage";
import DatasetsPage from "./pages/DatasetsPage";
import ClassifiersPage from "./pages/ClassifiersPage";
import RuntimePage from "./pages/RuntimePage";
import SuperclassHistogramPage from "./pages/SuperclassHistogramPage";
import FeatureAnalyticsPage from "./pages/FeatureAnalyticsPage";
import ValidationPage from "./pages/ValidationPage";
import AppHeader from "./components/AppHeader";
import { productAreaForPath } from "./productNavigation";

type Route =
  | { name: "operations" }
  | { name: "diagnostics" }
  | { name: "runtime" }
  | { name: "calibration" }
  | { name: "studio" }
  | { name: "validation" }
  | { name: "studio_report" }
  | { name: "datasets" }
  | { name: "classifiers" }
  | { name: "feature_analytics" }
  | { name: "superclass_histograms" }
  | { name: "take"; takeId: string };

function currentRoute(): Route {
  const path = window.location.pathname;
  if (path.startsWith("/takes/")) {
    return { name: "take", takeId: decodeURIComponent(path.replace("/takes/", "")) };
  }
  if (path === "/studio/report") {
    return { name: "studio_report" };
  }
  const area = productAreaForPath(path);
  return { name: area === "take" ? "operations" : area };
}

export default function App() {
  const route = currentRoute();
  const headerActive = route.name === "studio_report" ? "studio" : route.name;
  return (
    <>
      <AppHeader active={headerActive} />
      {route.name === "calibration" && <CalibrationPage />}
      {route.name === "studio" && <ProcessingLabPage />}
      {route.name === "validation" && <ValidationPage />}
      {route.name === "studio_report" && <PipelineWalkthroughPage />}
      {route.name === "datasets" && <DatasetsPage />}
      {route.name === "classifiers" && <ClassifiersPage />}
      {route.name === "feature_analytics" && <FeatureAnalyticsPage />}
      {route.name === "superclass_histograms" && <SuperclassHistogramPage />}
      {route.name === "runtime" && <RuntimePage />}
      {route.name === "diagnostics" && <DebugPage />}
      {route.name === "take" && <TakeDetailPage takeId={route.takeId} />}
      {route.name === "operations" && <OperatorPage />}
    </>
  );
}
