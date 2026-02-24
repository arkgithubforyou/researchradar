import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { getEntityList } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { formatCount } from "@/lib/utils";
import type { EntityType } from "@/lib/types";
import Spinner from "@/components/Spinner";

interface EntityListModalProps {
  entityType: EntityType;
  label: string;
  onClose: () => void;
}

export default function EntityListModal({
  entityType,
  label,
  onClose,
}: EntityListModalProps) {
  const { data, loading } = useAsync(
    () => getEntityList(entityType, { limit: 500 }),
    [entityType],
  );
  const backdropRef = useRef<HTMLDivElement>(null);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === backdropRef.current) onClose();
  };

  return (
    <div
      ref={backdropRef}
      onClick={handleBackdropClick}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4"
    >
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b">
          <h2 className="text-base font-semibold text-gray-900">
            {label}
            {data && (
              <span className="text-gray-400 font-normal ml-2">
                ({data.length})
              </span>
            )}
          </h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-3">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Spinner />
            </div>
          ) : data && data.length > 0 ? (
            <ul className="divide-y divide-gray-100">
              {data.map((item) => (
                <li
                  key={item.name}
                  className="flex items-center justify-between py-2.5"
                >
                  <span className="text-sm text-gray-700 truncate mr-3">
                    {item.name}
                  </span>
                  <span className="badge-gray text-xs flex-shrink-0">
                    {formatCount(item.count)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-400 text-center py-8">
              No entities found.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
