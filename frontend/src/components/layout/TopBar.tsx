import { 
  ShieldCheck,
  Zap
} from 'lucide-react';
import { cn } from '../../lib/utils';
import type { DashboardContext } from '../../types';

interface TopBarProps {
  currentModality: string;
  setCurrentModality: (m: string) => void;
  context: DashboardContext;
  productName: string;
  setProductName: (n: string) => void;
  timeRange: number;
  setTimeRange: (r: number) => void;
  latency?: number;
}

export default function TopBar({ 
  currentModality, 
  setCurrentModality, 
  context, 
  productName, 
  setProductName,
  timeRange,
  setTimeRange,
  latency
}: TopBarProps) {
  const modalities = ['reviews', 'sales', 'usage'];

  return (
    <header className="sticky top-0 z-50 w-full h-20 bg-base/80 backdrop-blur-xl border-b border-border-subtle p-6 flex items-center justify-between shadow-sm">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-cyan-400 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-blue-500/20">
            PF
          </div>
          <div className="text-left">
            <h1 className="text-lg font-display font-bold tracking-tight text-white leading-tight">Product Fatigue</h1>
            <p className="text-[10px] uppercase tracking-widest text-accent font-bold">Intelligence Console</p>
          </div>
        </div>

        <div className="h-8 w-px bg-border-subtle" />

        <div className="flex items-center gap-2 bg-black/20 p-1 rounded-xl border border-border-subtle">
          {modalities.map((m) => (
            <button
              key={m}
              onClick={() => setCurrentModality(m)}
              className={cn(
                "px-4 py-1.5 rounded-lg text-sm font-semibold transition-all duration-300",
                currentModality === m 
                  ? "bg-white text-base shadow-lg" 
                  : "text-text-muted hover:text-white"
              )}
            >
              {m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex flex-col gap-1 items-end">
          <span className="text-[10px] uppercase tracking-widest text-text-muted font-bold tracking-[0.2em]">Target Product</span>
          <div className="relative">
            <input 
              value={productName}
              onChange={(e) => setProductName(e.target.value)}
              className="bg-transparent border-none text-right font-medium text-white p-0 focus:ring-0 w-48 text-sm"
              placeholder="Enter product name..."
            />
          </div>
        </div>

        <div className="flex flex-col gap-1 items-end">
          <span className="text-[10px] uppercase tracking-widest text-text-muted font-bold tracking-[0.2em]">Time Window</span>
          <select 
            value={timeRange}
            onChange={(e) => setTimeRange(Number(e.target.value))}
            className="bg-transparent border-none text-right font-medium text-white p-0 focus:ring-0 text-sm cursor-pointer"
          >
            <option value={6} className="bg-base">6 Months</option>
            <option value={12} className="bg-base">12 Months</option>
            <option value={18} className="bg-base">18 Months</option>
            <option value={24} className="bg-base">24 Months</option>
          </select>
        </div>

        <div className="h-8 w-px bg-border-subtle mx-2" />

        <div className="flex items-center gap-4">
          <div className="flex flex-col items-end">
            <span className="text-[10px] uppercase tracking-widest text-text-muted font-bold tracking-[0.2em]">Model Engine</span>
            <span className="text-sm font-medium text-white flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-risk-low" />
              {context.model_versions[currentModality] || 'v1.0.4-prod'}
            </span>
          </div>
          <div className="flex flex-col items-end border-l border-border-subtle pl-4">
            <span className="text-[10px] uppercase tracking-widest text-text-muted font-bold tracking-[0.2em]">Latency</span>
            <span className="text-sm font-medium text-white flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-accent" />
              {latency ? `${latency}ms` : '---'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
