import { useState, useEffect } from 'react';
import { 
  RefreshCw 
} from 'lucide-react';
import type { DashboardContext, PredictionResult } from './types';

// Layout Components
import TopBar from './components/layout/TopBar';
import Sidebar from './components/layout/Sidebar';
import MainStage from './components/layout/MainStage';

const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

export default function App() {
  const [context, setContext] = useState<DashboardContext | null>(null);
  const [currentModality, setCurrentModality] = useState<string>('reviews');
  const [productName, setProductName] = useState('Manual Product');
  const [timeRange, setTimeRange] = useState(12);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [scenarioDelta, setScenarioDelta] = useState(0);

  // Fetch initial context
  useEffect(() => {
    fetch(`${API_BASE}/dashboard/api/context`)
      .then(res => res.json())
      .then(data => {
        setContext(data);
        // Initialize features with defaults
        const config = data.modalities[currentModality];
        if (config) {
          const defaults: Record<string, number> = {};
          config.fields.forEach((f: any) => {
            defaults[f.name] = f.default ?? 0;
          });
          setFeatures(defaults);
        }
      });
  }, []);

  // Update features when modality changes
  useEffect(() => {
    if (context) {
      const config = context.modalities[currentModality];
      if (config) {
        const defaults: Record<string, number> = {};
        config.fields.forEach((f: any) => {
          defaults[f.name] = f.default ?? 0;
        });
        setFeatures(defaults);
      }
    }
  }, [currentModality]);

  const runPrediction = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE}/dashboard/api/predict/${currentModality}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features,
          product_name: productName,
          time_range_months: timeRange,
          scenario_delta_pct: scenarioDelta,
          scenario_feature: context?.modalities[currentModality]?.scenario_feature
        })
      });
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      console.error('Prediction failed', err);
    } finally {
      setIsLoading(false);
    }
  };

  if (!context) return (
    <div className="min-h-screen bg-base flex flex-col items-center justify-center gap-4">
      <RefreshCw className="w-8 h-8 text-accent animate-spin" />
      <p className="text-text-muted font-medium animate-pulse">Initializing Dashboard Engine...</p>
    </div>
  );

  return (
    <div className="flex flex-col min-h-screen bg-base text-text-main selection:bg-accent/30 font-sans">
      <TopBar 
        currentModality={currentModality}
        setCurrentModality={setCurrentModality}
        context={context}
        productName={productName}
        setProductName={setProductName}
        timeRange={timeRange}
        setTimeRange={setTimeRange}
        latency={prediction?.model_health?.api_latency_ms}
      />
      
      <div className="flex flex-1 gap-8 p-8 max-w-[1600px] mx-auto w-full">
        <Sidebar 
          currentModality={currentModality}
          config={context.modalities[currentModality]}
          features={features}
          setFeatures={setFeatures}
          runPrediction={runPrediction}
          isLoading={isLoading}
          scenarioDelta={scenarioDelta}
          setScenarioDelta={setScenarioDelta}
        />
        
        <MainStage 
          prediction={prediction} 
          isLoading={isLoading}
          context={context}
        />
      </div>
    </div>
  );
}
