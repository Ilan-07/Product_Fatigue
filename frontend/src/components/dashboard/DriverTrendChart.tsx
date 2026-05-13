import { BarChart3 } from 'lucide-react';
import type { PredictionResult } from '../../types';

interface DriverTrendChartProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

export default function DriverTrendChart({ prediction, isLoading }: DriverTrendChartProps) {
  if (isLoading) {
    return (
      <div className="glass-panel h-[280px] animate-pulse border-border-subtle flex items-center justify-center">
        <p className="text-xs font-bold uppercase tracking-widest text-text-muted">Loading drivers...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-panel h-[280px] flex items-center justify-center border-dashed border-border-subtle">
        <p className="text-xs text-text-muted italic">Run an assessment to view fatigue drivers</p>
      </div>
    );
  }

  const shapEntries = Object.entries(prediction.prediction.shap_top5_features);
  if (shapEntries.length === 0) {
    return (
      <div className="glass-panel h-[280px] flex items-center justify-center border-dashed border-border-subtle">
        <p className="text-xs text-text-muted italic">No feature importance data available</p>
      </div>
    );
  }

  // Sort by absolute value descending
  const sorted = [...shapEntries].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbsVal = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.01);

  const barHeight = 32;
  const barGap = 12;
  const W = 600;
  const padL = 180;
  const padR = 60;
  const padT = 10;
  const padB = 10;
  const chartW = W - padL - padR;
  const centerX = padL + chartW / 2;
  const totalH = padT + sorted.length * (barHeight + barGap) + padB;

  return (
    <div className="glass-panel p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-accent" />
          <span className="text-xs font-bold uppercase tracking-widest text-text-muted">
            Top Fatigue Drivers
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-bold uppercase tracking-widest text-text-muted">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-risk-high inline-block rounded" /> Increases Risk
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-risk-low inline-block rounded" /> Decreases Risk
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${totalH}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
        {/* Center axis */}
        <line
          x1={centerX} y1={padT - 4}
          x2={centerX} y2={totalH - padB + 4}
          stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="4,3"
        />

        {sorted.map(([feature, value], i) => {
          const y = padT + i * (barHeight + barGap);
          const isPositive = value > 0;
          const normalizedW = (Math.abs(value) / maxAbsVal) * (chartW / 2);
          const barX = isPositive ? centerX : centerX - normalizedW;
          const color = isPositive ? '#ff4d4d' : '#00fa9a';
          const bgColor = isPositive ? 'rgba(255,77,77,0.06)' : 'rgba(0,250,154,0.06)';

          // Format feature name: replace underscores, capitalize
          const displayName = feature
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (c) => c.toUpperCase());

          return (
            <g key={feature}>
              {/* Feature label */}
              <text
                x={padL - 8} y={y + barHeight / 2 + 4}
                fill="rgba(255,255,255,0.7)" fontSize="10" textAnchor="end"
                fontWeight="500"
              >
                {displayName}
              </text>

              {/* Background */}
              <rect
                x={isPositive ? centerX : centerX - chartW / 2}
                y={y} width={chartW / 2} height={barHeight}
                rx="4" fill={bgColor}
              />

              {/* Value bar */}
              <rect x={barX} y={y} width={normalizedW} height={barHeight}
                rx="4" fill={color} opacity="0.7">
                <animate attributeName="width" from="0" to={normalizedW} dur="0.6s" fill="freeze" />
              </rect>

              {/* Value label */}
              <text
                x={isPositive ? centerX + normalizedW + 6 : centerX - normalizedW - 6}
                y={y + barHeight / 2 + 4}
                fill={color} fontSize="10" fontWeight="700" fontFamily="monospace"
                textAnchor={isPositive ? 'start' : 'end'}
              >
                {isPositive ? '+' : ''}{value.toFixed(3)}
              </text>
            </g>
          );
        })}

        {/* Axis labels */}
        <text x={padL} y={totalH - 2} fill="rgba(255,255,255,0.25)" fontSize="8" fontFamily="monospace">
          Protective
        </text>
        <text x={W - padR} y={totalH - 2} fill="rgba(255,255,255,0.25)" fontSize="8" textAnchor="end" fontFamily="monospace">
          Risk-Driving
        </text>
      </svg>
    </div>
  );
}
