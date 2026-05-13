import { 
  TrendingUp, 
  AlertTriangle,
  CheckCircle2,
  Activity
} from 'lucide-react';
import { cn } from '../../lib/utils';
import type { PredictionResult } from '../../types';

interface PrimaryMetricProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

export default function PrimaryMetric({ prediction, isLoading }: PrimaryMetricProps) {
  if (isLoading) {
    return (
      <div className="glass-panel h-[420px] flex flex-col items-center justify-center gap-6 animate-pulse border-accent/20">
        <div className="w-56 h-56 rounded-full border-8 border-accent/10 border-t-accent animate-spin" />
        <p className="text-sm font-bold uppercase tracking-widest text-accent">Recalculating Risk Vector...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-panel h-[420px] flex flex-col items-center justify-center gap-6 border-dashed border-border-subtle group">
        <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-accent/10 transition-all duration-500">
          <Activity className="w-8 h-8 text-text-muted group-hover:text-accent group-hover:scale-110 transition-all duration-500" />
        </div>
        <div className="text-center flex flex-col gap-2">
          <h3 className="text-lg font-display font-bold text-white tracking-tight">Awaiting Simulation</h3>
          <p className="max-w-[280px] text-xs leading-relaxed text-text-muted font-medium mx-auto italic opacity-80">
            Define your product parameters and run the assessment to generate the fatigue intelligence model.
          </p>
        </div>
      </div>
    );
  }

  const { risk_score, risk_band, natural_summary } = prediction;
  
  const scoreColor = 
    risk_band === 'High Fatigue' ? 'text-risk-high' : 
    risk_band === 'Moderate Fatigue' ? 'text-risk-med' : 
    'text-risk-low';

  const glowColor = 
    risk_band === 'High Fatigue' ? 'shadow-risk-high/40 border-risk-high/30' : 
    risk_band === 'Moderate Fatigue' ? 'shadow-risk-med/30 border-risk-med/30' : 
    'shadow-risk-low/20 border-risk-low/30';

  const arcColor = 
    risk_band === 'High Fatigue' ? '#ff4d4d' : 
    risk_band === 'Moderate Fatigue' ? '#fcc419' : 
    '#00fa9a';

  // Calculate SVG arc path for gauge
  const radius = 90;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (risk_score / 100) * circumference;

  return (
    <div className={cn("glass-panel h-[420px] overflow-hidden flex flex-col gap-6", glowColor)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-accent" />
          <span className="text-xs font-bold uppercase tracking-widest text-text-muted">Fatigue Risk Overview</span>
        </div>
        <div className={cn(
          "px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest flex items-center gap-1.5",
          risk_band === 'High Fatigue' ? "bg-risk-high/10 text-risk-high" : 
          risk_band === 'Moderate Fatigue' ? "bg-risk-med/10 text-risk-med" : 
          "bg-risk-low/10 text-risk-low"
        )}>
          {risk_band === 'High Fatigue' ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
          {risk_band}
        </div>
      </div>

      <div className="flex items-center gap-10 flex-1">
        <div className="relative w-52 h-52 flex items-center justify-center flex-shrink-0">
          <svg className="w-full h-full transform -rotate-90">
            <circle
              className="text-white/5"
              strokeWidth="12"
              stroke="currentColor"
              fill="transparent"
              r={radius}
              cx="104"
              cy="104"
            />
            <circle
              strokeWidth="12"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              stroke={arcColor}
              fill="transparent"
              r={radius}
              cx="104"
              cy="104"
              className="transition-all duration-1000 ease-out shadow-lg"
              style={{ filter: `drop-shadow(0 0 10px ${arcColor}44)` }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center mt-2">
            <span className={cn("text-6xl font-display font-bold tabular-nums tracking-tighter leading-none", scoreColor)}>
              {Math.round(risk_score)}
            </span>
            <span className="text-sm font-bold text-text-muted uppercase tracking-widest mt-1">/ 100</span>
          </div>
        </div>

        <div className="flex flex-col gap-6 max-w-sm">
          <div>
            <h2 className="text-2xl font-display font-bold text-white tracking-tight flex items-center gap-2">
              Assessment Summary
            </h2>
            <div className="h-1 w-12 bg-accent mt-2 rounded-full" />
          </div>
          
          <p className="text-base text-white/80 leading-relaxed font-medium">
            {natural_summary}
          </p>

          <div className="flex flex-wrap gap-3">
            {Object.entries(prediction.prediction.shap_top5_features).slice(0, 3).map(([feature, val]) => (
              <div key={feature} className="px-3 py-1.5 rounded-lg bg-white/5 border border-white/5 flex items-center gap-2">
                <span className="text-[10px] font-bold uppercase tracking-wider text-text-muted">{feature.replace('_', ' ')}</span>
                <span className={cn("text-[10px] font-bold", val > 0 ? "text-risk-high" : "text-risk-low")}>
                  {val > 0 ? '↑' : '↓'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-white/5">
        <div 
          className={cn("h-full transition-all duration-1000", 
            risk_band === 'High Fatigue' ? "bg-risk-high shadow-[0_0_15px_rgba(255,77,77,0.5)]" : 
            risk_band === 'Moderate Fatigue' ? "bg-risk-med shadow-[0_0_15px_rgba(252,196,25,0.5)]" : 
            "bg-risk-low shadow-[0_0_15px_rgba(0,250,154,0.5)]"
          )}
          style={{ width: `${prediction.completeness * 100}%` }}
        />
      </div>
    </div>
  );
}
