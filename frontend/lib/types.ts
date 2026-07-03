export type SeverityLevel = "low" | "moderate" | "high" | "unknown";

export interface DiseaseInfo {
  description: string;
  severity: string;
  recommendation: string;
  severity_level: SeverityLevel;
}

export interface ProbabilityItem {
  class_name: string;
  probability: number;
}

export interface PredictResponse {
  predicted_class: string;
  confidence: number;
  confidence_level: "high" | "medium" | "low" | "healthy";
  probabilities: ProbabilityItem[];
  disease_info: DiseaseInfo;
  is_healthy: boolean;
  model_mode: "checkpoint" | "demo";
  model_architecture: string;
}

export interface ClassItem {
  class_name: string;
  disease_info: DiseaseInfo;
}

export interface FilterOption {
  label: string;
  value: SeverityLevel | "all";
}