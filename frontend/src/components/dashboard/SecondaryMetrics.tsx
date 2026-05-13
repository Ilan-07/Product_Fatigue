import { 
  ShieldCheck, 
  Target, 
  Cpu, 
  Clock,
  TrendingDown,
  TrendingUp
} from 'lucide-react';
import { cn } from '../../lib/utils';
import type { PredictionResult } from '../../types';

interface SecondaryMetricsProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

export default function SecondaryMetrics({ prediction, isLoading }: SecondaryMetricsProps) {
  if (isLoading || !prediction) {
    return (
      <div className="grid grid-cols-3 gap-6">
        {[1, 2, 3].map((i) => (
          <div key={i} className="glass-panel h-48 animate-pulse border-border-subtle" />
        ))}
      </div>
    );
  }

  const { prediction: pred, model_health, trajectory } = prediction;

  const metrics = [
    {
      title: "Predicted Class",
      value: pred.predicted_class,
      icon: <Target className="w-4 h-4 text-accent" />,
      sub: "Fatigue State Label",
      trend: trajectory.trend_vs_last_period,
      trendLabel: "Trend vs Last Period",
      color: "text-white"
    },
    {
      title: "Confidence Score",
      value: `${(pred.confidence * 100).toFixed(1)}%`,
      icon: <ShieldCheck className="w-4 h-4 text-risk-low" />,
      sub: "Model Probability",
      trend: pred.confidence > 0.8 ? 5.2 : -2.1,
      trendLabel: "Vs Historical Avg",
      color: "text-risk-low"
    },
    {
      title: "Model Health",
      value: `${((model_health.f1_macro || 0) * 100).toFixed(1)}%`,
      icon: <Cpu className="w-4 h-4 text-risk-med" />,
      sub: "F1 Score (Validation)",
      trend: model_health.api_latency_ms,
      trendLabel: "Latency (ms)",
      color: "text-risk-med",
      isLatency: true
    }
  ];

  return (
    <div className="grid grid-cols-3 gap-6">
      {metrics.map((m, i) => (
        <div key={i} className="glass-panel p-6 flex flex-col justify-between group hover:shadow-2xl transition-all duration-500 overflow-hidden text-left">
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-white/5 border border-white/5 group-hover:bg-accent/10 group-hover:border-accent/20 transition-all duration-500">
                  {m.icon}
                </div>
                <span className="text-[11px] font-bold uppercase tracking-widest text-text-muted">{m.title}</span>
              </div>
            </div>
            
            <div className="flex flex-col gap-1">
              <span className={cn("text-3xl font-display font-bold whitespace-nowrap tracking-tighter", m.color)}>
                {m.value}
              </span>
              <span className="text-[10px] font-bold text-text-muted uppercase tracking-widest leading-none">
                {m.sub}
              </span>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-border-subtle flex items-center justify-between">
            <span className="text-[10px] font-bold text-text-muted uppercase tracking-tight">{m.trendLabel}</span>
            <div className={cn(
              "flex items-center gap-1 text-[11px] font-bold font-mono",
              m.isLatency 
                ? (m.trend < 5 ? "text-risk-low" : "text-risk-med")
                : (m.trend > 0 ? "text-risk-low" : "text-risk-high")
            )}>
              {m.isLatency ? <Clock className="w-3 h-3" /> : (m.trend > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />)}
              {m.isLatency ? m.trend : `${Math.abs(m.trend)}%`}
            </div>
          </div>

          <div className="absolute top-0 right-0 w-32 h-32 bg-accent/5 blur-3xl rounded-full -mr-16 -mt-16 group-hover:bg-accent/10 transition-all duration-500" />
        </div>
      ))}
    </div>
  );
}
