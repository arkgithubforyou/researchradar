import { Outlet, NavLink } from "react-router-dom";
import { Search, BookOpen, BarChart3, Radar, Info } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { to: "/", label: "Search", icon: Search, end: true },
  { to: "/browse", label: "Papers", icon: BookOpen },
  { to: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { to: "/about", label: "About", icon: Info },
] as const;

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Top nav ──────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            {/* Logo */}
            <NavLink
              to="/"
              className="flex items-center gap-2 text-brand-700 font-semibold text-lg select-none"
            >
              <Radar className="w-6 h-6" />
              <span className="hidden sm:inline">ResearchRadar</span>
            </NavLink>

            {/* Nav links */}
            <nav className="flex items-center gap-1">
              {NAV_ITEMS.map(({ to, label, icon: Icon, ...rest }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={"end" in rest}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
                      isActive
                        ? "bg-brand-50 text-brand-700"
                        : "text-gray-500 hover:text-gray-900 hover:bg-gray-100",
                    )
                  }
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{label}</span>
                </NavLink>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* ── Page content ─────────────────────────────────────────── */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* ── Footer ───────────────────────────────────────────────── */}
      <footer className="border-t bg-white py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-xs text-gray-400 text-center">
            ResearchRadar &mdash; RAG-powered NLP/ML research exploration.
            Hybrid retrieval + LLM generation over{" "}
            <span className="font-medium text-gray-500">ACL Anthology</span>{" "}
            papers.
          </p>
        </div>
      </footer>
    </div>
  );
}
