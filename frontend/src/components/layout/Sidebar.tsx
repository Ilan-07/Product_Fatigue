import type { Dispatch, SetStateAction } from 'react';
import { 
  Play, 
  Settings2, 
  Activity, 
  TrendingDown,
  TrendingUp,
  RefreshCw
} from 'lucide-react';
import { cn } from '../../lib/utils';
import type { ModalityConfig } from '../../types';

interface SidebarProps {
  currentModality: string;
  config: ModalityConfig;
  features: Record<string, number>;
  setFeatures: Dispatch<SetStateAction<Record<string, number>>>;
  runPrediction: () => void;
  isLoading: boolean;
  scenarioDelta: number;
  setScenarioDelta: (d: number) => void;
}

export default function Sidebar({ 
  currentModality, 
  config, 
  features, 
  setFeatures, 
  runPrediction, 
  isLoading,
  scenarioDelta,
  setScenarioDelta
}: SidebarProps) {
  const handleFeatureChange = (name: string, value: number) => {
    setFeatures(prev => ({ ...prev, [name]: value }));
  };

  const applySample = (sampleKey: string) => {
    const sample = config.samples[sampleKey];
    if (sample) {
      setFeatures(sample.features);
    }
  };

  return (
    <aside className="w-80 flex flex-col gap-8 flex-shrink-0 animate-in fade-in slide-in-from-left duration-700">
      <div className="flex flex-col gap-4 text-left">
        <button 
          onClick={runPrediction}
          disabled={isLoading}
          className="btn-primary w-full h-14 text-base font-bold flex items-center gap-3 shadow-xl"
        >
          {isLoading ? (
            <RefreshCw className="w-5 h-5 animate-spin" />
          ) : (
            <Play className="fill-current w-5 h-5" />
          )}
          {isLoading ? 'Processing...' : 'Run Assessment'}
        </button>

        <div className="grid grid-cols-2 gap-2">
          {Object.entries(config.samples).map(([key, sample]) => (
            <button
              key={key}
              onClick={() => applySample(key)}
              className="px-2 py-2 text-[9px] font-bold uppercase tracking-widest rounded-lg border border-border-subtle bg-white/5 hover:bg-white/10 text-text-muted hover:text-white transition-all text-center truncate"
              title={sample.label}
            >
              {sample.label}
            </button>
          ))}
        </div>
      </div>

      <div className="glass-panel p-5 flex flex-col gap-6 text-left">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-accent" />
            <span className="text-xs font-bold uppercase tracking-widest text-text-muted">Dynamic Inputs</span>
          </div>
          <span className="text-[10px] font-bold text-accent uppercase">{currentModality}</span>
        </div>

        <div className="flex flex-col gap-5 overflow-y-auto max-h-[500px] pr-1 -mr-1 custom-scrollbar">
          {config.fields.map(field => (
            <div key={field.name} className="flex flex-col gap-2">
              <div className="flex justify-between items-center px-1">
                <label className="text-[11px] font-semibold text-text-muted uppercase tracking-wide">{field.label}</label>
                <span className="text-xs font-mono font-bold text-white tabular-nums bg-white/5 px-2 py-0.5 rounded-md">
                  {features[field.name]?.toFixed(field.step?.toString().split('.')[1]?.length || 0)}
                </span>
              </div>
              <input 
                type="range"
                min={field.min}
                max={field.max}
                step={field.step}
                value={features[field.name] || 0}
                onChange={(e) => handleFeatureChange(field.name, Number(e.target.value))}
                className="w-full h-1.5 bg-black/40 rounded-lg appearance-none cursor-pointer accent-accent transition-all hover:bg-black/60 outline-none"
              />
            </div>
          ))}
        </div>
      </div>

      <div className="glass-panel p-5 flex flex-col gap-4 border-accent/20 bg-accent/[0.03] text-left">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-accent" />
            <span className="text-xs font-bold uppercase tracking-widest text-white">Scenario Simulation</span>
          </div>
          <span className={cn(
            "text-xs font-bold font-mono px-2 py-0.5 rounded-md",
            scenarioDelta > 0 ? "text-risk-low bg-risk-low/10" : scenarioDelta < 0 ? "text-risk-high bg-risk-high/10" : "text-text-muted bg-white/5"
          )}>
            {scenarioDelta > 0 ? '+' : ''}{scenarioDelta}%
          </span>
        </div>
        
        <p className="text-[10px] text-text-muted leading-relaxed font-bold">
          Simulate <span className="text-accent underline decoration-accent/30">{config.scenario_feature.replace('_', ' ')}</span> drift to forecast fatigue acceleration.
        </p>

        <input 
          type="range"
          min={-50}
          max={50}
          step={5}
          value={scenarioDelta}
          onChange={(e) => setScenarioDelta(Number(e.target.value))}
          className="w-full h-1.5 bg-black/40 rounded-lg appearance-none cursor-pointer accent-accent transition-all hover:bg-black/60 outline-none"
        />
        
        <div className="flex justify-between text-[10px] font-bold text-text-muted uppercase tracking-tighter pt-1">
          <div className="flex flex-col items-start gap-1">
            <TrendingDown className="w-3 h-3 text-risk-high" />
            <span>Heavy Drift</span>
          </div>
          <div className="flex flex-col items-center gap-1 opacity-40 uppercase tracking-widest text-[9px]">
            <span>Steady</span>
          </div>
          <div className="flex flex-col items-end gap-1">
            <TrendingUp className="w-3 h-3 text-risk-low" />
            <span>Recovery</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
