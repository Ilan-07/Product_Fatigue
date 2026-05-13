import { TrendingUp, AlertTriangle } from 'lucide-react';
import type { PredictionResult } from '../../types';

interface FatigueTrajectoryChartProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

export default function FatigueTrajectoryChart({ prediction, isLoading }: FatigueTrajectoryChartProps) {
  if (isLoading) {
    return (
      <div className="glass-panel h-[340px] animate-pulse border-border-subtle flex items-center justify-center">
        <p className="text-xs font-bold uppercase tracking-widest text-text-muted">Loading trajectory...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-panel h-[340px] flex items-center justify-center border-dashed border-border-subtle">
        <p className="text-xs text-text-muted italic">Run an assessment to view fatigue trajectory</p>
      </div>
    );
  }

  const { trajectory } = prediction;
  const { labels, fatigue, confidence, thresholds, events } = trajectory;
  const n = fatigue.length;
  if (n === 0) return null;

  // Chart dimensions
  const W = 700;
  const H = 220;
  const padL = 40;
  const padR = 20;
  const padT = 20;
  const padB = 40;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  const minY = 0;
  const maxY = 100;

  const xScale = (i: number) => padL + (i / Math.max(1, n - 1)) * chartW;
  const yScale = (v: number) => padT + chartH - ((v - minY) / (maxY - minY)) * chartH;

  // Fatigue line path
  const fatiguePath = fatigue
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`)
    .join(' ');

  // Confidence band (fatigue +/- scaled confidence)
  const bandWidth = 12;
  const upperBand = fatigue.map((v, i) => Math.min(100, v + bandWidth * (1 - confidence[i])));
  const lowerBand = fatigue.map((v, i) => Math.max(0, v - bandWidth * (1 - confidence[i])));

  const bandPath =
    upperBand.map((v, i) => `${i === 0 ? 'M' : 'L'}${xScale(i).toFixed(1)},${yScale(v).toFixed(1)}`).join(' ') +
    ' ' +
    lowerBand
      .map((v, i) => `L${xScale(n - 1 - i).toFixed(1)},${yScale(v).toFixed(1)}`)
      .reverse()
      .join(' ') +
    ' Z';

  // Threshold lines
  const healthyY = yScale(thresholds.healthy);
  const highY = yScale(thresholds.high);

  // Color of the final point
  const lastVal = fatigue[n - 1];
  const lineColor = lastVal >= thresholds.high ? '#ff4d4d' : lastVal >= thresholds.healthy ? '#fcc419' : '#00fa9a';

  return (
    <div className="glass-panel p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-accent" />
          <span className="text-xs font-bold uppercase tracking-widest text-text-muted">
            Fatigue Trajectory
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-bold uppercase tracking-widest text-text-muted">
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-risk-low inline-block rounded" /> Healthy
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-risk-med inline-block rounded" /> Moderate
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-0.5 bg-risk-high inline-block rounded" /> High
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={lineColor} stopOpacity="0.15" />
            <stop offset="100%" stopColor={lineColor} stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.3" />
            <stop offset="100%" stopColor={lineColor} />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((v) => (
          <g key={v}>
            <line
              x1={padL} y1={yScale(v)} x2={W - padR} y2={yScale(v)}
              stroke="rgba(255,255,255,0.06)" strokeWidth="1"
            />
            <text
              x={padL - 8} y={yScale(v) + 3}
              fill="rgba(255,255,255,0.3)" fontSize="9" textAnchor="end" fontFamily="monospace"
            >
              {v}
            </text>
          </g>
        ))}

        {/* Threshold zones */}
        {/* High fatigue zone */}
        <rect x={padL} y={padT} width={chartW} height={highY - padT}
          fill="rgba(255,77,77,0.04)" />
        {/* Healthy zone */}
        <rect x={padL} y={healthyY} width={chartW} height={padT + chartH - healthyY}
          fill="rgba(0,250,154,0.03)" />

        {/* Threshold lines */}
        <line x1={padL} y1={healthyY} x2={W - padR} y2={healthyY}
          stroke="#00fa9a" strokeWidth="1" strokeDasharray="6,4" opacity="0.5" />
        <text x={W - padR + 4} y={healthyY + 3} fill="#00fa9a" fontSize="8" opacity="0.6" fontFamily="monospace">
          Healthy
        </text>

        <line x1={padL} y1={highY} x2={W - padR} y2={highY}
          stroke="#ff4d4d" strokeWidth="1" strokeDasharray="6,4" opacity="0.5" />
        <text x={W - padR + 4} y={highY + 3} fill="#ff4d4d" fontSize="8" opacity="0.6" fontFamily="monospace">
          High
        </text>

        {/* Confidence band */}
        <path d={bandPath} fill="url(#bandGrad)" />

        {/* Main fatigue line */}
        <path d={fatiguePath} fill="none" stroke="url(#lineGrad)" strokeWidth="2.5"
          strokeLinecap="round" strokeLinejoin="round" />

        {/* Data points */}
        {fatigue.map((v, i) => {
          const cx = xScale(i);
          const cy = yScale(v);
          const ptColor = v >= thresholds.high ? '#ff4d4d' : v >= thresholds.healthy ? '#fcc419' : '#00fa9a';
          return (
            <g key={i}>
              <circle cx={cx} cy={cy} r="3" fill={ptColor} opacity="0.9" />
              {i === n - 1 && (
                <circle cx={cx} cy={cy} r="6" fill="none" stroke={ptColor} strokeWidth="1.5" opacity="0.5">
                  <animate attributeName="r" from="6" to="12" dur="2s" repeatCount="indefinite" />
                  <animate attributeName="opacity" from="0.5" to="0" dur="2s" repeatCount="indefinite" />
                </circle>
              )}
            </g>
          );
        })}

        {/* Event markers */}
        {events.map((evt, i) => {
          const monthIdx = Math.min(evt.month, n - 1);
          const cx = xScale(monthIdx);
          const cy = yScale(fatigue[monthIdx] ?? 50);
          return (
            <g key={`evt-${i}`}>
              <line x1={cx} y1={cy - 15} x2={cx} y2={cy - 5}
                stroke="#fcc419" strokeWidth="1" opacity="0.6" />
              <polygon
                points={`${cx - 4},${cy - 18} ${cx + 4},${cy - 18} ${cx},${cy - 12}`}
                fill="#fcc419" opacity="0.7"
              />
              <title>{evt.label}: {evt.detail}</title>
            </g>
          );
        })}

        {/* X-axis labels */}
        {labels.map((label, i) => (
          <text
            key={i}
            x={xScale(i)} y={H - 8}
            fill="rgba(255,255,255,0.35)" fontSize="9" textAnchor="middle" fontFamily="monospace"
          >
            {label}
          </text>
        ))}
      </svg>

      {/* Event legend */}
      {events.length > 0 && (
        <div className="flex gap-4 flex-wrap">
          {events.map((evt, i) => (
            <div key={i} className="flex items-center gap-2 text-[10px] text-text-muted">
              <AlertTriangle className="w-3 h-3 text-risk-med" />
              <span className="font-bold text-white/70">{evt.label}</span>
              <span className="italic">{evt.detail}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
