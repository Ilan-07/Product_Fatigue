import {
  AlertCircle,
  CheckCircle2,
  Zap,
  Activity,
  Layers,
  Sparkles,
  ArrowRight,
  RefreshCw,
  LineChart,
} from 'lucide-react';
import type { PredictionResult, DashboardContext } from '../../types';

// Components
import PrimaryMetric from '../dashboard/PrimaryMetric';
import SecondaryMetrics from '../dashboard/SecondaryMetrics';
import FatigueTrajectoryChart from '../dashboard/FatigueTrajectoryChart';
import BranchComparisonChart from '../dashboard/BranchComparisonChart';
import DriverTrendChart from '../dashboard/DriverTrendChart';
import ConfidenceGauge from '../dashboard/ConfidenceGauge';

interface MainStageProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
  context: DashboardContext;
}

export default function MainStage({ prediction, isLoading, context }: MainStageProps) {
  return (
    <main className="flex-1 flex flex-col gap-10 min-w-0 animate-in fade-in slide-in-from-right duration-700 text-left">
      {/* PRIMARY SECTION (FOCUS AREA) */}
      <section className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-display font-bold text-white tracking-tight flex items-center gap-2.5">
            <Activity className="w-5 h-5 text-accent" />
            Product Health Analysis
          </h2>
          <div className="text-[10px] uppercase tracking-widest text-text-muted font-bold flex items-center gap-2">
            Model Stability: <span className="text-risk-low">Optimum</span>
            <div className="w-1 h-1 rounded-full bg-risk-low shadow-[0_0_8px_#00fa9a]" />
          </div>
        </div>
        <PrimaryMetric prediction={prediction} isLoading={isLoading} />
      </section>

      {/* SECONDARY SECTION */}
      <section className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-display font-bold text-white tracking-tight flex items-center gap-2.5">
            <Layers className="w-5 h-5 text-accent" />
            Decision Parameters
          </h2>
        </div>
        <SecondaryMetrics prediction={prediction} isLoading={isLoading} />
      </section>

      {/* TRAJECTORY & CONFIDENCE SECTION */}
      <section className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-display font-bold text-white tracking-tight flex items-center gap-2.5">
            <LineChart className="w-5 h-5 text-accent" />
            Fatigue Intelligence
          </h2>
        </div>

        <FatigueTrajectoryChart prediction={prediction} isLoading={isLoading} />

        <div className="grid grid-cols-2 gap-6">
          <ConfidenceGauge prediction={prediction} isLoading={isLoading} />
          <DriverTrendChart prediction={prediction} isLoading={isLoading} />
        </div>

        <BranchComparisonChart prediction={prediction} isLoading={isLoading} />
      </section>

      {/* STRATEGIC INTERVENTIONS SECTION */}
      <section className="flex flex-col gap-6 opacity-80 hover:opacity-100 transition-opacity duration-500">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-display font-bold text-white tracking-tight flex items-center gap-2.5">
            <Zap className="w-5 h-5 text-accent" />
            Strategic Interventions
          </h2>
        </div>

        <div className="grid grid-cols-2 gap-8">
          <div className="glass-panel p-6 flex flex-col gap-6 border-risk-high/10 bg-risk-high/[0.01]">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-risk-high/10">
                <AlertCircle className="w-4 h-4 text-risk-high" />
              </div>
              <div className="flex flex-col text-left">
                <h3 className="text-sm font-bold text-white tracking-tight">Active Anomalies</h3>
                <p className="text-[10px] uppercase tracking-widest text-text-muted font-bold">Priority Alerts</p>
              </div>
            </div>

            <ul className="flex flex-col gap-3">
              {prediction?.alerts.length ? (
                prediction.alerts.map((alert, i) => (
                  <li key={i} className="flex gap-3 items-start p-3 rounded-xl bg-white/5 border border-white/5 text-xs text-white/70 font-medium leading-relaxed italic text-left">
                    <span className="w-1.5 h-1.5 rounded-full bg-risk-high mt-1.5 flex-shrink-0" />
                    {alert}
                  </li>
                ))
              ) : (
                <li className="flex gap-3 items-center p-3 rounded-xl bg-white/5 border border-dashed border-white/10 text-[11px] text-text-muted font-bold italic justify-center">
                  System state stable - No active alerts.
                </li>
              )}
            </ul>
          </div>

          <div className="glass-panel p-6 flex flex-col gap-6 border-accent/10 bg-accent/[0.01]">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-accent/10">
                <Sparkles className="w-4 h-4 text-accent" />
              </div>
              <div className="flex flex-col text-left">
                <h3 className="text-sm font-bold text-white tracking-tight">Strategic Actions</h3>
                <p className="text-[10px] uppercase tracking-widest text-text-muted font-bold">Actionable Insights</p>
              </div>
            </div>

            <ul className="flex flex-col gap-3">
              {prediction?.recommended_actions.length ? (
                prediction.recommended_actions.map((action, i) => (
                  <li key={i} className="flex items-center justify-between p-3 rounded-xl bg-accent/5 border border-accent/10 group cursor-pointer hover:bg-accent/10 transition-colors">
                    <div className="flex gap-3 items-center">
                      <div className="p-1 rounded-md bg-accent/20">
                        <CheckCircle2 className="w-3 h-3 text-accent" />
                      </div>
                      <span className="text-xs text-white font-medium">{action}</span>
                    </div>
                    <ArrowRight className="w-3 h-3 text-accent opacity-0 group-hover:opacity-100 transition-opacity" />
                  </li>
                ))
              ) : (
                <li className="flex gap-3 items-center p-3 rounded-xl bg-white/5 border border-dashed border-white/10 text-[11px] text-text-muted font-bold italic justify-center">
                  Recommendations pending assessment...
                </li>
              )}
            </ul>
          </div>
        </div>
      </section>

      {/* FOOTER METADATA */}
      <footer className="pt-8 mt-4 border-t border-border-subtle flex items-center justify-between opacity-50">
        <div className="flex items-center gap-8 text-[10px] uppercase tracking-[0.2em] font-bold text-text-muted">
          <div className="flex items-center gap-2">
            <Activity className="w-3 h-3" />
            Backend Protocol: HTTP/2
          </div>
          <div className="flex items-center gap-2">
            <RefreshCw className="w-3 h-3" />
            Last Retrained: {context.last_retrained || 'Unknown'}
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-[0.2em] font-bold text-text-muted flex items-center gap-2">
          Engine Version: {context.model_versions.reviews || 'v1.0.0'}
          <div className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
        </div>
      </footer>
    </main>
  );
}
