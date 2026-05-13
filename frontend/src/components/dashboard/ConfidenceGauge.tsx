import { ShieldCheck, AlertTriangle, Eye } from 'lucide-react';
import { cn } from '../../lib/utils';
import type { PredictionResult } from '../../types';

interface ConfidenceGaugeProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

export default function ConfidenceGauge({ prediction, isLoading }: ConfidenceGaugeProps) {
  if (isLoading) {
    return (
      <div className="glass-panel h-[260px] animate-pulse border-border-subtle flex items-center justify-center">
        <p className="text-xs font-bold uppercase tracking-widest text-text-muted">Calibrating...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-panel h-[260px] flex items-center justify-center border-dashed border-border-subtle">
        <p className="text-xs text-text-muted italic">Run an assessment to view confidence</p>
      </div>
    );
  }

  const confidence = prediction.prediction.confidence;
  const riskBand = prediction.risk_band;

  // Confidence band thresholds
  const isHigh = confidence >= 0.80;
  const isMedium = confidence >= 0.60 && confidence < 0.80;
  const isLow = confidence < 0.60;

  // Uncertainty flag: low confidence or high fatigue with low confidence
  const uncertaintyFlag = isLow || (riskBand === 'High Fatigue' && confidence < 0.70);

  const confidencePct = confidence * 100;

  // Gauge arc parameters (semicircle)
  const cx = 100;
  const cy = 105;
  const r = 80;
  const startAngle = Math.PI; // 180 degrees (left)
  const endAngle = 0; // 0 degrees (right)
  const sweepAngle = startAngle - endAngle;
  const valueAngle = startAngle - (confidence * sweepAngle);

  const arcPath = (start: number, end: number) => {
    const x1 = cx + r * Math.cos(start);
    const y1 = cy - r * Math.sin(start);
    const x2 = cx + r * Math.cos(end);
    const y2 = cy - r * Math.sin(end);
    const largeArc = Math.abs(start - end) > Math.PI ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
  };

  // Confidence color
  const gaugeColor = isHigh ? '#00fa9a' : isMedium ? '#fcc419' : '#ff4d4d';
  const bandLabel = isHigh ? 'HIGH' : isMedium ? 'MEDIUM' : 'LOW';

  // Needle tip position
  const needleX = cx + (r - 10) * Math.cos(valueAngle);
  const needleY = cy - (r - 10) * Math.sin(valueAngle);

  return (
    <div className="glass-panel p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-accent" />
          <span className="text-xs font-bold uppercase tracking-widest text-text-muted">
            Confidence Gauge
          </span>
        </div>
        {uncertaintyFlag && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-risk-high/10 text-risk-high">
            <AlertTriangle className="w-3 h-3" />
            <span className="text-[10px] font-bold uppercase tracking-widest">Uncertain</span>
          </div>
        )}
      </div>

      <div className="flex items-center gap-8">
        <svg viewBox="0 0 200 130" className="w-48 h-auto flex-shrink-0">
          {/* Background arc segments: Low | Medium | High */}
          {/* Low (0-60%): red zone */}
          <path d={arcPath(Math.PI, Math.PI * 0.4)} fill="none"
            stroke="rgba(255,77,77,0.15)" strokeWidth="14" strokeLinecap="round" />
          {/* Medium (60-80%): yellow zone */}
          <path d={arcPath(Math.PI * 0.4, Math.PI * 0.2)} fill="none"
            stroke="rgba(252,196,25,0.15)" strokeWidth="14" strokeLinecap="round" />
          {/* High (80-100%): green zone */}
          <path d={arcPath(Math.PI * 0.2, 0)} fill="none"
            stroke="rgba(0,250,154,0.15)" strokeWidth="14" strokeLinecap="round" />

          {/* Active arc (filled to current confidence) */}
          <path d={arcPath(Math.PI, valueAngle)} fill="none"
            stroke={gaugeColor} strokeWidth="14" strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${gaugeColor}44)` }}
          />

          {/* Needle */}
          <line x1={cx} y1={cy} x2={needleX} y2={needleY}
            stroke="white" strokeWidth="2" strokeLinecap="round" opacity="0.8" />
          <circle cx={cx} cy={cy} r="4" fill="white" opacity="0.6" />

          {/* Center value */}
          <text x={cx} y={cy + 25} textAnchor="middle" fill="white"
            fontSize="22" fontWeight="700" fontFamily="monospace">
            {confidencePct.toFixed(1)}%
          </text>

          {/* Band label */}
          <text x={cx} y={cy + 40} textAnchor="middle" fill={gaugeColor}
            fontSize="9" fontWeight="700" letterSpacing="0.15em">
            {bandLabel} CONFIDENCE
          </text>

          {/* Scale labels */}
          <text x={15} y={cy + 10} fill="rgba(255,255,255,0.25)" fontSize="8" fontFamily="monospace">0%</text>
          <text x={180} y={cy + 10} fill="rgba(255,255,255,0.25)" fontSize="8" fontFamily="monospace" textAnchor="end">100%</text>
          <text x={cx} y={20} fill="rgba(255,255,255,0.2)" fontSize="7" fontFamily="monospace" textAnchor="middle">50%</text>
        </svg>

        <div className="flex flex-col gap-3 flex-1">
          {/* Confidence breakdown */}
          <div className="flex flex-col gap-2">
            <h4 className="text-[11px] font-bold uppercase tracking-widest text-text-muted">Threshold Analysis</h4>

            <div className="flex flex-col gap-1.5">
              {[
                { label: 'High Confidence', range: '80-100%', color: '#00fa9a', active: isHigh },
                { label: 'Medium Confidence', range: '60-80%', color: '#fcc419', active: isMedium },
                { label: 'Low / Uncertain', range: '0-60%', color: '#ff4d4d', active: isLow },
              ].map((tier) => (
                <div key={tier.label} className={cn(
                  "flex items-center justify-between px-3 py-1.5 rounded-lg transition-all",
                  tier.active ? "bg-white/5 border border-white/10" : "opacity-40"
                )}>
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: tier.color }} />
                    <span className="text-[10px] font-bold text-white/80">{tier.label}</span>
                  </div>
                  <span className="text-[10px] font-mono font-bold" style={{ color: tier.color }}>
                    {tier.range}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Uncertainty warning */}
          {uncertaintyFlag && (
            <div className="p-3 rounded-xl bg-risk-high/5 border border-risk-high/10 flex items-start gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-risk-high flex-shrink-0 mt-0.5" />
              <p className="text-[10px] text-risk-high/80 leading-relaxed font-medium">
                Prediction confidence is below the reliability threshold.
                Consider manual review before acting on this result.
              </p>
            </div>
          )}

          {!uncertaintyFlag && (
            <div className="p-3 rounded-xl bg-risk-low/5 border border-risk-low/10 flex items-start gap-2">
              <ShieldCheck className="w-3.5 h-3.5 text-risk-low flex-shrink-0 mt-0.5" />
              <p className="text-[10px] text-risk-low/80 leading-relaxed font-medium">
                Prediction meets confidence requirements. Automated decision-making is supported.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
