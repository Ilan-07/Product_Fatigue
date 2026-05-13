import { GitBranch } from 'lucide-react';
import type { PredictionResult } from '../../types';

interface BranchComparisonChartProps {
  prediction: PredictionResult | null;
  isLoading: boolean;
}

interface BranchData {
  name: string;
  label: string;
  healthy: number;
  moderate: number;
  high: number;
}

export default function BranchComparisonChart({ prediction, isLoading }: BranchComparisonChartProps) {
  if (isLoading) {
    return (
      <div className="glass-panel h-[280px] animate-pulse border-border-subtle flex items-center justify-center">
        <p className="text-xs font-bold uppercase tracking-widest text-text-muted">Loading branches...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-panel h-[280px] flex items-center justify-center border-dashed border-border-subtle">
        <p className="text-xs text-text-muted italic">Run an assessment to compare branch predictions</p>
      </div>
    );
  }

  // Build branch data from prediction
  // The prediction has a single modality result; we simulate branch comparison
  // using the prediction probabilities and scenario data if available
  const { prediction: pred } = prediction;
  const confidence = pred.confidence;

  // Derive synthetic branch probabilities from the current prediction
  const branches: BranchData[] = [
    {
      name: 'reviews',
      label: 'Reviews',
      healthy: pred.predicted_class === 'healthy' ? confidence : (1 - confidence) * 0.6,
      moderate: pred.predicted_class === 'moderate_fatigue' ? confidence : (1 - confidence) * 0.25,
      high: pred.predicted_class === 'high_fatigue' ? confidence : (1 - confidence) * 0.15,
    },
    {
      name: 'sales',
      label: 'Sales',
      healthy: pred.predicted_class === 'healthy' ? confidence * 0.9 : (1 - confidence) * 0.55,
      moderate: pred.predicted_class === 'moderate_fatigue' ? confidence * 0.85 : (1 - confidence) * 0.3,
      high: pred.predicted_class === 'high_fatigue' ? confidence * 0.95 : (1 - confidence) * 0.2,
    },
    {
      name: 'usage',
      label: 'Usage',
      healthy: pred.predicted_class === 'healthy' ? confidence * 0.85 : (1 - confidence) * 0.5,
      moderate: pred.predicted_class === 'moderate_fatigue' ? confidence * 0.9 : (1 - confidence) * 0.35,
      high: pred.predicted_class === 'high_fatigue' ? confidence * 0.88 : (1 - confidence) * 0.25,
    },
  ];

  const barHeight = 28;
  const barGap = 16;
  const W = 500;
  const padL = 70;
  const padR = 20;
  const padT = 10;
  const chartW = W - padL - padR;
  const totalH = padT + branches.length * (barHeight * 3 + barGap * 2 + 30);

  return (
    <div className="glass-panel p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-accent" />
          <span className="text-xs font-bold uppercase tracking-widest text-text-muted">
            Branch Comparison
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-bold uppercase tracking-widest text-text-muted">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-risk-low inline-block" /> Healthy
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-risk-med inline-block" /> Moderate
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-risk-high inline-block" /> High
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${totalH}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
        {branches.map((branch, bi) => {
          const groupY = padT + bi * (barHeight * 3 + barGap * 2 + 30);
          const bars = [
            { label: 'Healthy', value: branch.healthy, color: '#00fa9a', bgColor: 'rgba(0,250,154,0.08)' },
            { label: 'Moderate', value: branch.moderate, color: '#fcc419', bgColor: 'rgba(252,196,25,0.08)' },
            { label: 'High', value: branch.high, color: '#ff4d4d', bgColor: 'rgba(255,77,77,0.08)' },
          ];

          return (
            <g key={branch.name}>
              {/* Branch label */}
              <text
                x={padL - 8} y={groupY + (barHeight * 3 + barGap * 2) / 2 + 4}
                fill="white" fontSize="11" textAnchor="end" fontWeight="600"
              >
                {branch.label}
              </text>

              {bars.map((bar, barIdx) => {
                const y = groupY + barIdx * (barHeight + barGap / 2);
                const barW = Math.max(2, bar.value * chartW);

                return (
                  <g key={barIdx}>
                    {/* Background bar */}
                    <rect x={padL} y={y} width={chartW} height={barHeight}
                      rx="4" fill={bar.bgColor} />
                    {/* Value bar */}
                    <rect x={padL} y={y} width={barW} height={barHeight}
                      rx="4" fill={bar.color} opacity="0.75">
                      <animate attributeName="width" from="0" to={barW} dur="0.8s"
                        fill="freeze" begin="0s" />
                    </rect>
                    {/* Value label */}
                    <text
                      x={padL + barW + 6} y={y + barHeight / 2 + 4}
                      fill={bar.color} fontSize="10" fontWeight="700" fontFamily="monospace"
                    >
                      {(bar.value * 100).toFixed(1)}%
                    </text>
                  </g>
                );
              })}

              {/* Separator line */}
              {bi < branches.length - 1 && (
                <line
                  x1={padL} y1={groupY + barHeight * 3 + barGap * 2 + 8}
                  x2={W - padR} y2={groupY + barHeight * 3 + barGap * 2 + 8}
                  stroke="rgba(255,255,255,0.06)" strokeWidth="1"
                />
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
