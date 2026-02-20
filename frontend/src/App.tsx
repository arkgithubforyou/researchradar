import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import SearchPage from "./pages/SearchPage";
import BrowsePage from "./pages/BrowsePage";
import PaperPage from "./pages/PaperPage";
import DashboardPage from "./pages/DashboardPage";
import AboutPage from "./pages/AboutPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<SearchPage />} />
        <Route path="browse" element={<BrowsePage />} />
        <Route path="paper/:paperId" element={<PaperPage />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="about" element={<AboutPage />} />
      </Route>
    </Routes>
  );
}
