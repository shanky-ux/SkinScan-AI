import type { FilterOption } from "./types";

export const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const filterOptions: FilterOption[] = [
  { label: "All", value: "all" },
  { label: "Low", value: "low" },
  { label: "Moderate", value: "moderate" },
  { label: "High", value: "high" },
];

export const pipelineSteps = [
  { title: "Upload", detail: "Drag, drop, or capture a skin image" },
  { title: "Preprocess", detail: "Resize, normalize, and enhance contrast" },
  { title: "CNN Inference", detail: "Run the ResNet/EfficientNet classifier" },
  { title: "Prediction", detail: "Surface the most likely condition and confidence" },
];

export const featureCards = [
  {
    title: "Preprocessing pipeline",
    detail: "Quality checks, enhancement, and TTA mirror the training-time image flow.",
  },
  {
    title: "Model-driven inference",
    detail: "FastAPI wraps the PyTorch model and returns structured prediction output.",
  },
  {
    title: "Medical-tech UX",
    detail: "Glassmorphism panels, animated charts, and a dark premium surface.",
  },
];

export const classFallback = [
  {
    class_name: "Actinic keratosis",
    disease_info: {
      description: "Precancerous skin lesions caused by sun damage",
      severity: "Moderate risk - can develop into cancer",
      recommendation: "Medical evaluation and treatment recommended",
      severity_level: "moderate" as const,
    },
  },
  {
    class_name: "Atopic Dermatitis",
    disease_info: {
      description: "Chronic inflammatory skin condition (eczema)",
      severity: "Low to moderate risk",
      recommendation: "Dermatologist consultation for treatment plan",
      severity_level: "low" as const,
    },
  },
  {
    class_name: "Benign keratosis",
    disease_info: {
      description: "Non-cancerous skin growths",
      severity: "Low risk",
      recommendation: "Regular monitoring recommended",
      severity_level: "low" as const,
    },
  },
  {
    class_name: "Dermatofibroma",
    disease_info: {
      description: "Benign fibrous skin tumor",
      severity: "Low risk",
      recommendation: "Usually no treatment needed unless bothersome",
      severity_level: "low" as const,
    },
  },
  {
    class_name: "Melanocytic nevus",
    disease_info: {
      description: "Common benign skin growths (moles)",
      severity: "Generally benign",
      recommendation: "Monitor for changes in size, color, or shape",
      severity_level: "low" as const,
    },
  },
  {
    class_name: "Melanoma",
    disease_info: {
      description: "A serious form of skin cancer",
      severity: "High risk - requires immediate medical attention",
      recommendation: "Consult a dermatologist immediately",
      severity_level: "high" as const,
    },
  },
  {
    class_name: "Squamous cell carcinoma",
    disease_info: {
      description: "Second most common type of skin cancer",
      severity: "Moderate to high risk",
      recommendation: "Immediate dermatologist consultation required",
      severity_level: "high" as const,
    },
  },
  {
    class_name: "Tinea Ringworm Candidiasis",
    disease_info: {
      description: "Fungal skin infections",
      severity: "Low risk but contagious",
      recommendation: "Antifungal treatment and medical consultation",
      severity_level: "low" as const,
    },
  },
  {
    class_name: "Vascular lesion",
    disease_info: {
      description: "Lesions involving blood vessels in the skin",
      severity: "Generally low risk",
      recommendation: "Medical evaluation for proper diagnosis",
      severity_level: "low" as const,
    },
  },
];