export interface ModalityField {
  name: string;
  label: string;
  type: string;
  step?: number;
  min?: number;
  max?: number;
  default?: number;
}

export interface Sample {
  label: string;
  features: Record<string, number>;
}

export interface ModalityConfig {
  scenario_feature: string;
  fields: ModalityField[];
  samples: Record<string, Sample>;
}

export interface ModalityCard {
  modality: string;
  label: string;
  champion_model: string;
  model_version: string;
  f1_macro: number | null;
  roc_auc: number | null;
  accuracy: number | null;
  cluster_metrics: Record<string, unknown>;
}

export interface FatigueRate {
  modality: string;
  rate: number;
  label?: string;
}

export interface KeyMetric {
  label: string;
  value: number | string;
  unit?: string;
}

export interface DashboardContext {
  modalities: Record<string, ModalityConfig>;
  modality_cards: ModalityCard[];
  fatigue_rates: FatigueRate[];
  key_metrics: KeyMetric[];
  last_retrained: string | null;
  model_versions: Record<string, string>;
  api_status: {
    loaded_models: string[];
    model_versions: Record<string, string>;
  };
}

export interface PredictionResult {
  risk_score: number;
  risk_band: 'Healthy' | 'Moderate Fatigue' | 'High Fatigue';
  prediction: {
    predicted_class: string;
    confidence: number;
    shap_top5_features: Record<string, number>;
    cluster_id: number | null;
  };
  trajectory: {
    labels: string[];
    fatigue: number[];
    confidence: number[];
    thresholds: { healthy: number; high: number };
    trend_vs_last_period: number;
    events: { month: number; label: string; detail: string }[];
  };
  model_health: {
    f1_macro: number | null;
    roc_auc_ovr_macro: number | null;
    cv_test_gap: number | null;
    api_latency_ms: number;
  };
  alerts: string[];
  recommended_actions: string[];
  natural_summary: string;
  completeness: number;
}
