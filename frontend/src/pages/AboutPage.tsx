import { useState } from "react";
import { Radar, Mail, Eye, EyeOff } from "lucide-react";

export default function AboutPage() {
  const [showEmail, setShowEmail] = useState(false);

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="card p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-brand-100 flex items-center justify-center">
            <Radar className="w-5 h-5 text-brand-700" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">
            About ResearchRadar
          </h1>
        </div>

        <div className="space-y-4 text-sm text-gray-600 leading-relaxed">
          <p>
            Tracking and finding research papers is hard for researchers,
            especially in AI and ML where we have seen an exponential growth in
            the number of publications. ResearchRadar helps researchers
            discover, explore, and stay current with the literature through
            RAG-powered search and analytics over papers from the ACL Anthology
            and beyond.
          </p>
          <p>
            The system combines hybrid retrieval (dense vectors + BM25
            sparse retrieval) with LLM-powered answer generation, so you can
            ask natural-language questions and get grounded answers backed by
            real publications.
          </p>
        </div>

        <hr className="my-6 border-gray-100" />

        <div>
          <h2 className="text-sm font-semibold text-gray-900 mb-3">Contact</h2>
          <p className="text-sm text-gray-600 mb-3">Fangzhou Zhai</p>
          <button
            onClick={() => setShowEmail((s) => !s)}
            className="btn-secondary text-xs inline-flex items-center gap-1.5"
          >
            <Mail className="w-3.5 h-3.5" />
            {showEmail ? (
              <>
                Arkcertif@gmail.com
                <EyeOff className="w-3.5 h-3.5 ml-1" />
              </>
            ) : (
              <>
                Show email
                <Eye className="w-3.5 h-3.5 ml-1" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
